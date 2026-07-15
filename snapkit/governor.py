"""
Harmony Governor — Layer 2 of the triadic cognitive architecture.

Wires snapkit-v2 primitives into a real-time feedback loop:
  - BeatGrid as the system clock (tempo derived from physical sensors)
  - spectral_summary per channel as the FEP friction metric (Φ)
  - TemporalConnectome for coupled/decoupled agent detection
  - FluxTensorMIDI as the inter-agent event bus

The governor does not think. It measures friction and triggers the
Executive when friction exceeds the deadband tolerance.

Architecture: see docs/ARCHITECTURE_OF_HARMONY.md

Zero external dependencies. stdlib only. Python ≥ 3.10.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Deque, Dict, List, Optional, Tuple

from snapkit.spectral import spectral_summary, SpectralSummary
from snapkit.connectome import TemporalConnectome, ConnectomeResult
from snapkit.temporal import BeatGrid, TemporalSnap, TemporalResult
from snapkit.midi import FluxTensorMIDI, MIDIEvent, MIDIEventType, TempoMap


class FrictionLevel(IntEnum):
    """Cognitive friction severity."""
    HARMONY = 0       # Φ well within deadband — agent is in flow
    STRAIN = 1        # Φ approaching boundary — model is degrading
    SURPRISE = 2      # Φ exceeded deadband — model has failed, Executive needed


@dataclass(frozen=True, slots=True)
class FrictionAlarm:
    """Alarm fired when a channel's friction exceeds the deadband."""
    channel: int
    room_name: str
    level: FrictionLevel
    phi: float
    entropy: float
    hurst: float
    latency_ms: float
    tick: int
    timestamp: float
    message: str = ""


@dataclass
class ChannelState:
    """Per-channel rolling state for friction computation."""
    __slots__ = (
        'name', 'channel', 'window_size',
        'predictions', 'actuals', 'latencies',
        'phi_history', 'last_summary', 'last_update_tick',
        'deadband', 'hurst_floor',
        'sensor_lo', 'sensor_hi',
        'is_surprised_latched',
    )

    name: str
    channel: int
    window_size: int
    predictions: Deque[float]
    actuals: Deque[float]
    latencies: Deque[float]
    phi_history: Deque[float]
    last_summary: Optional[SpectralSummary]
    last_update_tick: int
    deadband: float
    hurst_floor: float
    sensor_lo: float
    sensor_hi: float
    is_surprised_latched: bool

    def __init__(
        self,
        name: str,
        channel: int,
        window_size: int = 128,
        deadband: float = 2.0,
        hurst_floor: float = 0.45,
        sensor_lo: float = 0.0,
        sensor_hi: float = 1.0,
    ):
        self.name = name
        self.channel = channel
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.phi_history = deque(maxlen=window_size)
        self.last_summary = None
        self.last_update_tick = 0
        self.deadband = deadband
        self.hurst_floor = hurst_floor
        self.sensor_lo = sensor_lo
        self.sensor_hi = sensor_hi
        # Alarm latch — fires once per surprise episode, not every tick
        # (fixes the bug where sustained alarms flood the Executive with
        # duplicate events for the same underlying alarm)
        self.is_surprised_latched = False

    @property
    def phi(self) -> float:
        """Current cognitive friction (Φ)."""
        if not self.phi_history:
            return 0.0
        return self.phi_history[-1]

    @property
    def is_in_harmony(self) -> bool:
        return self.phi < self.deadband * 0.7

    @property
    def is_strained(self) -> bool:
        return self.deadband * 0.7 <= self.phi and self.phi < self.deadband

    @property
    def is_surprised(self) -> bool:
        return self.phi >= self.deadband

    def record(
        self,
        prediction: float,
        actual: float,
        latency_ms: float,
        tick: int,
        alpha: float = 0.6,
        beta: float = 0.3,
        gamma: float = 0.1,
        delta: float = 0.4,
    ) -> float:
        """Record a prediction/actual pair and compute friction.

        Returns the instantaneous Φ value.

        Φ = α·H + β·L + γ·hurst_penalty + δ·error_magnitude

        Defaults chosen so coefficients (α=0.6, β=0.3, γ=0.1, δ=0.4) sum
        to ≤1.4 when components are normalised — informative, not magic.
        """
        # NaN/Inf guard — non-finite predictions crash MIDI export and confuse averages
        if (
            prediction != prediction or actual != actual
            or prediction in (float('inf'), float('-inf'))
            or actual in (float('inf'), float('-inf'))
        ):
            phi = float('nan')
            self.phi_history.append(phi)
            return phi

        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.latencies.append(latency_ms)
        self.last_update_tick = tick

        # Compute prediction error
        error = abs(prediction - actual)

        # Compute friction from the rolling window
        if len(self.actuals) < 4:
            phi = error  # Not enough data for spectral analysis
            self.phi_history.append(phi)
            return phi

        # Spectral analysis on the prediction error stream
        errors = [p - a for p, a in zip(self.predictions, self.actuals)]
        errors = [e for e in errors if e == e and abs(e) != float('inf')]
        if len(errors) < 4:
            phi = error
            self.phi_history.append(phi)
            return phi

        self.last_summary = spectral_summary(errors, bins=min(10, len(errors) // 2))

        # Normalized entropy (0-1 range)
        max_entropy = math.log2(min(10, len(errors) // 2)) if len(errors) >= 4 else 1.0
        norm_entropy = self.last_summary.entropy_bits / max_entropy if max_entropy > 0 else 0.0

        # Direct error magnitude (normalized) — catches steady drift
        # that low entropy misses (constant error = low entropy but high friction)
        recent_errors = errors[-min(8, len(errors)):]
        mean_abs_error = sum(abs(e) for e in recent_errors) / len(recent_errors)
        norm_error = min(mean_abs_error / 10.0, 1.0)

        # Hurst penalty: if H drops below floor, the model has lost structure
        hurst_penalty = 0.0
        if self.last_summary.hurst < self.hurst_floor:
            hurst_penalty = (self.hurst_floor - self.last_summary.hurst) * 2.0

        # Latency component (normalized) — use LATEST latency not the average,
        # so a sudden spike isn't diluted by history (Kimi bug 9-style fix)
        latest_latency = self.latencies[-1] if self.latencies else 0.0
        norm_latency = min(latest_latency / 1000.0, 1.0)

        # Φ = α·H + β·L + γ·hurst_penalty + δ·error_magnitude
        phi = (
            alpha * norm_entropy
            + beta * norm_latency
            + gamma * hurst_penalty
            + delta * norm_error
        )
        self.phi_history.append(phi)

        return phi


class HarmonyGovernor:
    """Layer 2: The FEP friction monitor and alarm router.

    The governor runs the beat grid, collects per-channel telemetry,
    and fires FrictionAlarms when agents exceed their deadband.

    Usage:
        governor = HarmonyGovernor(
            tempo_map=TempoMap(ticks_per_beat=480, initial_bpm=120),
        )
        governor.register_channel("helm", channel=0, deadband=2.0)
        governor.register_channel("nav", channel=1, deadband=1.5)

        # Each beat:
        governor.tick(tick=0)
        governor.record_observation("helm", prediction=0.3, actual=0.35, latency_ms=12)
        governor.record_observation("nav", prediction=180.0, actual=182.0, latency_ms=8)

        alarms = governor.alarms  # List[FrictionAlarm]
    """

    __slots__ = (
        '_channels', '_flux', '_tempo', '_beat_grid',
        '_temporal_snap', '_connectome',
        '_alarms', '_tick', '_sustained_threshold',
        '_sustained_counts', '_on_alarm',
    )

    def __init__(
        self,
        tempo_map: Optional[TempoMap] = None,
        beat_period: float = 1.0,
        sustained_threshold: int = 3,
        on_alarm: Optional[Callable[[FrictionAlarm], None]] = None,
    ):
        self._tempo: TempoMap = tempo_map or TempoMap()
        self._flux: FluxTensorMIDI = FluxTensorMIDI(self._tempo)
        self._beat_grid: BeatGrid = BeatGrid(period=beat_period)
        self._temporal_snap: TemporalSnap = TemporalSnap(self._beat_grid, tolerance=0.1)
        self._connectome: TemporalConnectome = TemporalConnectome(threshold=0.3, max_lag=8)
        self._channels: Dict[str, ChannelState] = {}
        self._alarms: List[FrictionAlarm] = []
        self._tick: int = 0
        self._sustained_threshold: int = sustained_threshold
        self._sustained_counts: Dict[str, int] = {}
        self._on_alarm: Optional[Callable[[FrictionAlarm], None]] = on_alarm

    def register_channel(
        self,
        name: str,
        channel: int,
        deadband: float = 2.0,
        hurst_floor: float = 0.45,
        window_size: int = 128,
        sensor_lo: float = 0.0,
        sensor_hi: float = 1.0,
    ) -> ChannelState:
        """Register a new agent channel for monitoring.

        Args:
            sensor_lo: Lower bound of the sensor value range (for MIDI mapping).
            sensor_hi: Upper bound of the sensor value range (for MIDI mapping).
                The default (0..1) covers normalized sensors. For heading,
                pass sensor_lo=0, sensor_hi=360.0 so MIDI pitch carries
                information across the full sensor range.
        """
        if name in self._channels:
            raise ValueError(f"channel '{name}' already registered")
        self._flux.add_room(name, channel=channel)
        state = ChannelState(
            name=name, channel=channel,
            window_size=window_size, deadband=deadband,
            hurst_floor=hurst_floor,
        )
        state.sensor_lo = sensor_lo
        state.sensor_hi = sensor_hi
        self._channels[name] = state
        self._sustained_counts[name] = 0
        return state

    def tick(self, tick: Optional[int] = None) -> None:
        """Advance the governor by one beat."""
        self._tick = tick if tick is not None else self._tick + 1

    def record_observation(
        self,
        channel_name: str,
        prediction: float,
        actual: float,
        latency_ms: float = 0.0,
    ) -> float:
        """Record a prediction/actual pair for a channel.

        Returns the instantaneous Φ for that channel.
        """
        if channel_name not in self._channels:
            raise KeyError(f"channel '{channel_name}' not registered")

        state = self._channels[channel_name]
        phi = state.record(prediction, actual, latency_ms, self._tick)

        # Map actual → MIDI note using this channel's configured sensor range.
        # Without this, a heading of e.g. 180° maps to note=22860 (saturated to 127),
        # making every pitch identical. With sensor_lo/sensor_hi (e.g. 0..360),
        # the note carries actual information across the sensor's full scale.
        span = state.sensor_hi - state.sensor_lo
        if span > 0 and actual == actual and abs(actual) != float('inf'):
            norm = max(0.0, min(1.0, (actual - state.sensor_lo) / span))
            note = int(round(norm * 127))
        else:
            note = 64  # neutral middle

        # NaN-safe velocity: high when φ is small, low when it spikes
        if phi == phi and abs(phi) != float('inf'):
            velocity = max(1, min(127, int(round(127 * (1.0 - min(phi / 3.0, 1.0))))))
        else:
            velocity = 1

        self._flux.note_on(
            channel_name, tick=self._tick,
            note=note, velocity=velocity,
        )

        # Check for alarm conditions
        self._check_alarms(state)

        return phi

    def _check_alarms(self, state: ChannelState) -> None:
        """Check if a channel's friction warrants an alarm.

        Alarm handling uses a latch: once a SURPRISE episode has fired,
        we don't fire another alarm until the channel returns to harmony.
        This counts *episodes*, not *surprised ticks* — a sustained
        50-tick surprise episode is one alarm, not fifty.
        """
        if state.is_surprised:
            self._sustained_counts[state.name] += 1
            if (
                self._sustained_counts[state.name] >= self._sustained_threshold
                and not state.is_surprised_latched
            ):
                alarm = FrictionAlarm(
                    channel=state.channel,
                    room_name=state.name,
                    level=FrictionLevel.SURPRISE,
                    phi=state.phi,
                    entropy=state.last_summary.entropy_bits if state.last_summary else 0.0,
                    hurst=state.last_summary.hurst if state.last_summary else 0.5,
                    latency_ms=sum(state.latencies) / max(len(state.latencies), 1),
                    tick=self._tick,
                    timestamp=time.time(),
                    message=f"Channel '{state.name}' friction Φ={state.phi:.3f} "
                            f"exceeded deadband {state.deadband}",
                )
                self._alarms.append(alarm)
                state.is_surprised_latched = True
                if self._on_alarm:
                    self._on_alarm(alarm)
        elif state.is_strained:
            # Strain degrades, but doesn't break the latch — wait for harmony
            self._sustained_counts[state.name] = max(0, self._sustained_counts[state.name] - 1)
        else:
            # In harmony — decay sustained count AND release the latch
            self._sustained_counts[state.name] = max(0, self._sustained_counts[state.name] - 2)
            state.is_surprised_latched = False

    def update_connectome(self) -> Optional[ConnectomeResult]:
        """Analyze coupling between channels.

        Call this periodically (every N ticks) to detect whether
        agents that should be resonating have decoupled.

        The result is cached on `self._connectome` so downstream consumers
        (like DiagnosticEngine) can read coupling without rebuilding it.
        """
        if len(self._channels) < 2:
            self._connectome_last_result = None
            return None

        # Reuse the instance connectome: it tracks coupling history across calls
        # (the previous implementation rebuilt it from scratch each time, which
        # discarded state needed for cross-episode cascade detection).
        self._connectome._traces.clear()
        for name, state in self._channels.items():
            if len(state.actuals) >= 5:
                self._connectome.add_room(name, list(state.actuals))

        if len(self._connectome._traces) < 2:
            self._connectome_last_result = None
            return None

        result = self._connectome.analyze()
        self._connectome_last_result = result
        return result

    @property
    def connectome_result(self) -> Optional[ConnectomeResult]:
        """Last computed connectome result (None if never computed)."""
        return getattr(self, "_connectome_last_result", None)

    @property
    def alarms(self) -> List[FrictionAlarm]:
        """All alarms fired since last clear."""
        return list(self._alarms)

    @property
    def unacknowledged_alarms(self) -> List[FrictionAlarm]:
        """Alarms at SURPRISE level that need Executive attention."""
        return [a for a in self._alarms if a.level == FrictionLevel.SURPRISE]

    def clear_alarms(self) -> None:
        """Acknowledge and clear all alarms."""
        self._alarms.clear()

    @property
    def tick_count(self) -> int:
        return self._tick

    @property
    def channels(self) -> List[str]:
        return list(self._channels.keys())

    def channel_state(self, name: str) -> ChannelState:
        return self._channels[name]

    def global_phi(self) -> float:
        """System-wide friction metric — mean Φ across all channels."""
        if not self._channels:
            return 0.0
        return sum(s.phi for s in self._channels.values()) / len(self._channels)

    def system_state(self) -> Dict:
        """Snapshot of the entire governor state."""
        return {
            'tick': self._tick,
            'global_phi': self.global_phi(),
            'channels': {
                name: {
                    'phi': s.phi,
                    'harmony': s.is_in_harmony,
                    'strained': s.is_strained,
                    'surprised': s.is_surprised,
                    'deadband': s.deadband,
                    'samples': len(s.actuals),
                    'hurst': s.last_summary.hurst if s.last_summary else None,
                    'entropy': s.last_summary.entropy_bits if s.last_summary else None,
                }
                for name, s in self._channels.items()
            },
            'active_alarms': len(self.unacknowledged_alarms),
            'total_alarms': len(self._alarms),
        }
