"""
MIDI I/O Bridge — Connects physical sensors and agent decisions to the
FluxTensorMIDI bus and Harmony Governor.

Supports:
  - ESP32 sensor ingestion via serial/WebSocket → Control Change messages
  - NMEA 2000 bus monitoring → structured sensor channels
  - Agent decision output → Note On/Off messages with velocity = confidence
  - BeatGrid derivation from physical IMU data (the hull sets the beat)

When `mido` is available, can also interface with real MIDI hardware
(synthesizers, MIDI interfaces, USB-MIDI devices). Falls back to
internal event routing when mido is not installed.

Zero external dependencies (mido optional). stdlib only. Python ≥ 3.10.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

from snapkit.midi import (
    FluxTensorMIDI, MIDIEvent, MIDIEventType, Room, TempoMap,
)
from snapkit.governor import HarmonyGovernor
from snapkit.temporal import BeatGrid
from snapkit.spectral import spectral_summary


# Standard MIDI mapping for maritime sensors
SENSOR_CC_MAP = {
    "rudder_angle": 1,       # CC #1 — continuous 0-127
    "rpm": 2,                # CC #2
    "heading": 3,            # CC #3
    "speed": 4,              # CC #4 (knots)
    "depth": 5,              # CC #5 (meters)
    "wind_speed": 6,         # CC #6
    "wind_direction": 7,     # CC #7
    "water_temp": 8,         # CC #8
    "air_temp": 9,           # CC #9
    "barometric": 10,        # CC #10
    "bilge_water": 11,       # CC #11
    "battery_voltage": 12,   # CC #12
    "fuel_level": 13,        # CC #13
    "roll_angle": 14,        # CC #14 (from IMU)
    "pitch_angle": 15,       # CC #15 (from IMU)
}

# Note mapping for agent actions
ACTION_NOTE_MAP = {
    "rudder_left": 0,
    "rudder_right": 1,
    "throttle_up": 2,
    "throttle_down": 3,
    "engine_start": 4,
    "engine_stop": 5,
    "pump_on": 6,
    "pump_off": 7,
    "alarm_acknowledge": 8,
    "gear_deploy": 9,
    "gear_retrieve": 10,
    "nav_light_on": 11,
    "nav_light_off": 12,
    "anchor_drop": 13,
    "anchor_raise": 14,
    "mayday": 15,
}


@dataclass
class SensorReading:
    """A single sensor reading mapped to MIDI."""
    sensor_name: str
    cc_number: int
    raw_value: float        # Physical value (e.g., 180.0 degrees)
    midi_value: int          # 0-127
    timestamp: float
    source: str = "unknown"  # "nmea2000", "esp32", "opencv", etc.


@dataclass
class AgentAction:
    """An agent decision mapped to MIDI."""
    agent_name: str
    action_name: str
    note_number: int
    velocity: int            # 0-127, confidence × urgency
    timestamp: float
    description: str = ""


class SensorMapper:
    """Maps physical sensor values to/from MIDI 0-127 range."""

    @staticmethod
    def to_midi(value: float, lo: float, hi: float) -> int:
        """Map a physical value to MIDI 0-127."""
        if hi == lo:
            return 0
        normalized = (value - lo) / (hi - lo)
        return max(0, min(127, int(normalized * 127)))

    @staticmethod
    def from_midi(midi_val: int, lo: float, hi: float) -> float:
        """Map MIDI 0-127 back to physical value."""
        return lo + (midi_val / 127.0) * (hi - lo)

    @staticmethod
    def angle_to_midi(degrees: float) -> int:
        """Map an angle (0-360) to MIDI."""
        return max(0, min(127, int((degrees % 360) / 360.0 * 127)))

    @staticmethod
    def midi_to_angle(midi_val: int) -> float:
        """Map MIDI back to angle (0-360)."""
        return (midi_val / 127.0) * 360.0


class TempoDeriver:
    """Derives the BeatGrid tempo from physical IMU data.

    The hull sets the beat. The encounter frequency of waves and the
    vessel's roll period determine the system tempo. When the sea state
    changes, the tempo changes, and every agent must re-phase-lock.
    """

    __slots__ = (
        '_roll_history', '_window_size',
        '_last_bpm', '_last_period', '_stability',
    )

    def __init__(self, window_size: int = 256):
        self._roll_history: Deque[float] = deque(maxlen=window_size)
        self._window_size = window_size
        self._last_bpm: float = 60.0
        self._last_period: float = 1.0
        self._stability: float = 0.0

    def feed(self, roll_angle: float, timestamp: Optional[float] = None) -> None:
        """Feed a roll angle reading from the IMU."""
        self._roll_history.append(roll_angle)

        if len(self._roll_history) >= 32:
            self._derive_tempo()

    def _derive_tempo(self) -> None:
        """Analyze roll history for dominant period via autocorrelation.

        For a periodic signal, autocorrelation starts high (lag 0),
        drops, then rises again at the signal's period. We look for
        the first significant peak AFTER the initial drop.
        """
        data = list(self._roll_history)
        n = len(data)
        if n < 16:
            return

        mean = sum(data) / n
        centered = [x - mean for x in data]
        r0 = sum(x * x for x in centered) / n
        if r0 < 1e-10:
            self._stability = 1.0
            return

        # Compute autocorrelation
        max_lag = min(n // 2, 64)
        acf = []
        for lag in range(max_lag):
            rk = sum(centered[t] * centered[t + lag] for t in range(n - lag)) / n
            rk /= r0
            acf.append(rk)

        # Find the first zero-crossing (end of the initial decay)
        # Then find the first peak after that
        zero_cross = 0
        for i in range(1, len(acf)):
            if acf[i-1] > 0 > acf[i]:
                zero_cross = i
                break

        # Find first peak after zero-crossing
        best_lag = 0
        best_corr = 0.0
        if zero_cross > 0:
            for lag in range(zero_cross, max_lag):
                # Local peak: higher than neighbors
                if lag + 1 < len(acf):
                    if acf[lag] > acf[lag - 1] and acf[lag] >= acf[lag + 1]:
                        if acf[lag] > 0.2:
                            best_lag = lag
                            best_corr = acf[lag]
                            break
                elif acf[lag] > best_corr and acf[lag] > 0.2:
                    best_lag = lag
                    best_corr = acf[lag]

        if best_lag > 0:
            sample_rate = 10.0  # Hz (assumed IMU rate)
            period_seconds = best_lag / sample_rate
            if 0.5 <= period_seconds <= 20.0:
                self._last_period = period_seconds
                self._last_bpm = 60.0 / period_seconds
                self._stability = best_corr

    @property
    def bpm(self) -> float:
        return self._last_bpm

    @property
    def period(self) -> float:
        return self._last_period

    @property
    def stability(self) -> float:
        """How stable the tempo is (0-1, from autocorrelation peak)."""
        return self._stability

    @property
    def beat_grid(self) -> BeatGrid:
        """Get a BeatGrid synced to the hull's tempo."""
        return BeatGrid(period=self._last_period)

    def state(self) -> Dict:
        return {
            'bpm': self._last_bpm,
            'period_seconds': self._last_period,
            'stability': self._stability,
            'samples': len(self._roll_history),
        }


class MIDIBridge:
    """Connects physical sensors and agent decisions to the MIDI bus.

    This is the I/O layer that translates between the physical world
    (ESP32 sensors, NMEA 2000, camera feeds) and the MIDI protocol
    that carries agent state across the network.

    Usage:
        bridge = MIDIBridge(governor=governor)
        bridge.register_sensor("heading", lo=0, hi=360, source="nmea2000")

        # Feed sensor data:
        bridge.feed_sensor("heading", 182.5)

        # Agent makes a decision:
        bridge.agent_action("helm", "rudder_left", velocity=100, confidence=0.9)

        # Derive tempo from IMU:
        bridge.feed_roll(5.2)  # 5.2 degrees of roll
        print(bridge.tempo_deriver.bpm)
    """

    __slots__ = (
        '_governor', '_flux', '_sensors',
        '_tempo_deriver', '_mido_available',
        '_midi_out', '_action_history',
    )

    def __init__(
        self,
        governor: Optional[HarmonyGovernor] = None,
        flux: Optional[FluxTensorMIDI] = None,
    ):
        self._governor = governor
        self._flux = flux or governor._flux if governor else FluxTensorMIDI()
        self._sensors: Dict[str, Dict] = {}
        self._tempo_deriver = TempoDeriver()
        self._action_history: Deque[AgentAction] = deque(maxlen=256)

        # Check for mido (optional, for real MIDI hardware)
        try:
            import mido
            self._mido_available = True
            self._midi_out = None
        except ImportError:
            self._mido_available = False
            self._midi_out = None

    def register_sensor(
        self,
        name: str,
        lo: float = 0.0,
        hi: float = 360.0,
        source: str = "unknown",
        cc_number: Optional[int] = None,
    ) -> None:
        """Register a physical sensor for MIDI ingestion."""
        if cc_number is None:
            cc_number = SENSOR_CC_MAP.get(name, 16 + len(self._sensors))
        self._sensors[name] = {
            'lo': lo, 'hi': hi, 'source': source,
            'cc_number': cc_number,
            'last_value': None, 'last_midi': 0,
            'history': deque(maxlen=128),
        }
        # Also register as a room in the MIDI flux if not already present
        if self._governor and name not in self._governor.channels:
            self._governor.register_channel(
                name, channel=cc_number % 16,
            )
        elif name not in self._flux.rooms:
            self._flux.add_room(name, channel=cc_number % 16)

    def feed_sensor(
        self,
        name: str,
        value: float,
        timestamp: Optional[float] = None,
    ) -> Optional[SensorReading]:
        """Feed a physical sensor reading into the MIDI bus."""
        if name not in self._sensors:
            raise KeyError(f"sensor '{name}' not registered")

        cfg = self._sensors[name]
        ts = timestamp or time.time()
        midi_val = SensorMapper.to_midi(value, cfg['lo'], cfg['hi'])

        reading = SensorReading(
            sensor_name=name,
            cc_number=cfg['cc_number'],
            raw_value=value,
            midi_value=midi_val,
            timestamp=ts,
            source=cfg['source'],
        )

        cfg['last_value'] = value
        cfg['last_midi'] = midi_val
        cfg['history'].append(value)

        # Emit CC event on the MIDI bus
        if self._governor:
            tick = self._governor.tick_count
            self._flux.note_on(
                name, tick=tick,
                note=midi_val,
                velocity=64,  # Neutral velocity for sensor data
            )

        # Send to real MIDI hardware if available
        if self._mido_available and self._midi_out:
            import mido
            msg = mido.Message(
                'control_change',
                control=cfg['cc_number'],
                value=midi_val,
            )
            self._midi_out.send(msg)

        return reading

    def feed_roll(self, roll_degrees: float) -> float:
        """Feed IMU roll data to derive hull tempo.

        Returns the current derived BPM.
        """
        self._tempo_deriver.feed(roll_degrees)
        return self._tempo_deriver.bpm

    def agent_action(
        self,
        agent_name: str,
        action_name: str,
        velocity: int = 100,
        confidence: float = 1.0,
        description: str = "",
    ) -> AgentAction:
        """Record an agent's decision as a MIDI Note On event.

        Velocity encodes the agent's confidence × urgency.
        High velocity = high confidence (harmony).
        Low velocity = low confidence (friction).
        """
        note = ACTION_NOTE_MAP.get(action_name, 16)
        # Blend velocity with confidence
        effective_velocity = max(1, min(127, int(velocity * confidence)))

        action = AgentAction(
            agent_name=agent_name,
            action_name=action_name,
            note_number=note,
            velocity=effective_velocity,
            timestamp=time.time(),
            description=description,
        )
        self._action_history.append(action)

        # Emit on MIDI bus
        if self._governor:
            tick = self._governor.tick_count
            self._flux.note_on(
                agent_name, tick=tick,
                note=note,
                velocity=effective_velocity,
            )

        # Real MIDI hardware
        if self._mido_available and self._midi_out:
            import mido
            channel = 0  # Would map from agent_name
            msg = mido.Message(
                'note_on',
                channel=channel,
                note=note,
                velocity=effective_velocity,
            )
            self._midi_out.send(msg)

        return action

    def open_midi_port(self, port_name: Optional[str] = None) -> bool:
        """Open a real MIDI output port (requires mido)."""
        if not self._mido_available:
            return False
        import mido
        try:
            if port_name is None:
                ports = mido.get_output_names()
                if not ports:
                    return False
                port_name = ports[0]
            self._midi_out = mido.open_output(port_name)
            return True
        except Exception:
            return False

    def playback_events(self) -> List[MIDIEvent]:
        """Render all MIDI events in tick order (for playback/debugging)."""
        return self._flux.render()

    def export_midi_file(self, filename: str) -> bool:
        """Export the session as a standard MIDI file (requires mido)."""
        if not self._mido_available:
            return False
        import mido
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        events = self.playback_events()
        prev_tick = 0
        for event in events:
            delta = event.tick - prev_tick
            msg = mido.Message(
                'note_on' if event.event_type == MIDIEventType.NOTE_ON else 'note_off',
                channel=event.channel,
                note=event.value,
                velocity=event.velocity,
                time=delta,
            )
            track.append(msg)
            prev_tick = event.tick

        mid.save(filename)
        return True

    @property
    def tempo_deriver(self) -> TempoDeriver:
        return self._tempo_deriver

    @property
    def sensors(self) -> List[str]:
        return list(self._sensors.keys())

    @property
    def mido_available(self) -> bool:
        return self._mido_available

    def state(self) -> Dict:
        """Snapshot of the MIDI bridge state."""
        return {
            'sensors': {
                name: {
                    'cc': cfg['cc_number'],
                    'source': cfg['source'],
                    'last_value': cfg['last_value'],
                    'range': [cfg['lo'], cfg['hi']],
                }
                for name, cfg in self._sensors.items()
            },
            'tempo': self._tempo_deriver.state(),
            'midi_hardware': self._midi_out is not None,
            'mido_available': self._mido_available,
            'total_actions': len(self._action_history),
            'total_events': self._flux.event_count,
        }
