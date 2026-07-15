"""
Fleet Coordinator — Multi-vessel harmony governance.

Extends the single-vessel triadic architecture to a fleet of vessels.
Each vessel runs its own Harmony Governor. The Fleet Coordinator:

  1. Aggregates per-vessel harmony state across the fleet
  2. Detects fleet-level patterns (coordinated drift, cascading failures)
  3. Routes Executive attention to the vessel that needs it most
  4. Shares learned models between vessels on similar routes
  5. Maintains a fleet-wide MIDI bus with one channel per vessel

Each vessel is a track in the score. The fleet is the orchestra.

Zero external dependencies. stdlib only. Python ≥ 3.10.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Deque, Dict, List, Optional, Tuple

from snapkit.governor import HarmonyGovernor, FrictionLevel, FrictionAlarm
from snapkit.connectome import TemporalConnectome, ConnectomeResult
from snapkit.midi import FluxTensorMIDI, TempoMap


class FleetAlert(IntEnum):
    """Fleet-level alert types."""
    SINGLE_VESSEL = 0    # One vessel in trouble
    COORDINATED = 1      # Multiple vessels failing simultaneously
    CASCADE = 2          # Failures spreading from one vessel to others
    SILENT_FLEET = 3     # All vessels stopped reporting (communication failure)
    DRIFT = 4            # Fleet heading diverging from coordinated plan


@dataclass(frozen=True, slots=True)
class VesselSnapshot:
    """Point-in-time snapshot of one vessel's harmony state."""
    vessel_id: str
    channel: int          # MIDI channel
    phi: float            # Cognitive friction
    bpm: float            # Hull-derived tempo
    heading: float        # Current heading
    heading_error: float  # Deviation from target
    sea_state: str        # calm/moderate/rough
    is_in_harmony: bool
    is_strained: bool
    is_surprised: bool
    last_update: float    # Unix timestamp
    alarms: int           # Unacknowledged alarm count


@dataclass(frozen=True, slots=True)
class FleetEvent:
    """A fleet-level event detected by the coordinator."""
    alert_type: FleetAlert
    vessels: List[str]   # Affected vessel IDs
    description: str
    timestamp: float
    severity: float       # 0-1


class FleetCoordinator:
    """Coordinates harmony governance across multiple vessels.

    Each vessel runs its own Harmony Governor locally. The Fleet
    Coordinator receives periodic snapshots and watches for patterns
    that no single vessel can see.

    In a real fleet deployment, snapshots arrive via VHF/satellite/sideband.
    The coordinator runs on shore or on the mothership.

    Usage:
        fleet = FleetCoordinator()
        fleet.register_vessel("f/v_northstar", channel=0)
        fleet.register_vessel("f/v_aurora", channel=1)
        fleet.register_vessel("f/v_kestrel", channel=2)

        # Each heartbeat, vessels report in:
        fleet.update_vessel(VesselSnapshot(
            vessel_id="f/v_northstar", channel=0,
            phi=0.3, bpm=24.0, heading=180.0, heading_error=2.0,
            sea_state="moderate", is_in_harmony=True,
            is_strained=False, is_surprised=False,
            last_update=time.time(), alarms=0,
        ))

        # Check for fleet-level events:
        events = fleet.detect_events()
    """

    def __init__(
        self,
        heartbeat_timeout: float = 300.0,  # 5 min without update = silent
        drift_threshold: float = 15.0,      # Degrees of heading divergence
    ):
        self._vessels: Dict[str, Dict] = {}
        self._flux: FluxTensorMIDI = FluxTensorMIDI(
            TempoMap(ticks_per_beat=480, initial_bpm=120)
        )
        self._history: Deque[VesselSnapshot] = deque(maxlen=10000)
        self._events: List[FleetEvent] = []
        self._heartbeat_timeout = heartbeat_timeout
        self._drift_threshold = drift_threshold
        self._vessel_connectome = TemporalConnectome(threshold=0.3, max_lag=5)
        self._vessel_activity: Dict[str, List[float]] = {}

    def register_vessel(
        self,
        vessel_id: str,
        channel: int,
        target_heading: float = 0.0,
    ) -> None:
        """Register a vessel in the fleet."""
        if vessel_id in self._vessels:
            raise ValueError(f"vessel '{vessel_id}' already registered")
        self._vessels[vessel_id] = {
            'channel': channel,
            'target_heading': target_heading,
            'last_snapshot': None,
            'phi_history': deque(maxlen=256),
            'heading_history': deque(maxlen=256),
            'alarm_count': 0,
        }
        self._flux.add_room(vessel_id, channel=channel)
        self._vessel_activity[vessel_id] = []

    def update_vessel(self, snapshot: VesselSnapshot) -> None:
        """Receive a harmony snapshot from a vessel."""
        if snapshot.vessel_id not in self._vessels:
            return  # Unknown vessel, ignore

        vdata = self._vessels[snapshot.vessel_id]
        vdata['last_snapshot'] = snapshot
        vdata['phi_history'].append(snapshot.phi)
        vdata['heading_history'].append(snapshot.heading)
        vdata['alarm_count'] = snapshot.alarms

        self._history.append(snapshot)

        # Track activity for connectome
        self._vessel_activity[snapshot.vessel_id] = \
            list(vdata['phi_history'])[-64:]

        # Emit on MIDI bus
        velocity = max(1, min(127, int(127 * (1.0 - min(snapshot.phi / 3.0, 1.0)))))
        self._flux.note_on(
            snapshot.vessel_id,
            tick=int(time.time() * 10),  # 0.1s resolution
            note=int(snapshot.heading / 360.0 * 127),
            velocity=velocity,
        )

    def detect_events(self) -> List[FleetEvent]:
        """Detect fleet-level events.

        This is the fleet Executive — it sees patterns no single vessel can.
        """
        events: List[FleetEvent] = []
        now = time.time()

        active_vessels = {
            vid: v for vid, v in self._vessels.items()
            if v['last_snapshot'] and (now - v['last_snapshot'].last_update < self._heartbeat_timeout)
        }

        if not active_vessels:
            events.append(FleetEvent(
                alert_type=FleetAlert.SILENT_FLEET,
                vessels=list(self._vessels.keys()),
                description="No vessels reporting — possible communication failure",
                timestamp=now, severity=1.0,
            ))
            return events

        # Check for single-vessel distress
        surprised = [
            v['last_snapshot'] for v in active_vessels.values()
            if v['last_snapshot'].is_surprised
        ]
        if surprised:
            events.append(FleetEvent(
                alert_type=FleetAlert.SINGLE_VESSEL,
                vessels=[s.vessel_id for s in surprised],
                description=f"{len(surprised)} vessel(s) in SURPRISE state",
                timestamp=now, severity=0.8,
            ))

        # Check for coordinated failure (2+ vessels surprised simultaneously)
        if len(surprised) >= 2:
            events.append(FleetEvent(
                alert_type=FleetAlert.COORDINATED,
                vessels=[s.vessel_id for s in surprised],
                description=f"Coordinated failure: {len(surprised)} vessels surprised simultaneously — possible weather event",
                timestamp=now, severity=0.9,
            ))

        # Check for heading drift (fleet diverging)
        headings = [
            v['last_snapshot'].heading for v in active_vessels.values()
        ]
        if len(headings) >= 2:
            heading_range = max(headings) - min(headings)
            if heading_range > self._drift_threshold:  # configurable drift threshold
                events.append(FleetEvent(
                    alert_type=FleetAlert.DRIFT,
                    vessels=list(active_vessels.keys()),
                    description=f"Fleet heading spread: {heading_range:.0f}° — vessels diverging",
                    timestamp=now, severity=min(1.0, heading_range / 90.0),
                ))

        # Check for cascade (one vessel's alarms correlate with others failing)
        if len(active_vessels) >= 3:
            cascade = self._detect_cascade(active_vessels)
            if cascade:
                events.append(cascade)

        # Check heartbeat timeouts
        silent = [
            vid for vid, v in self._vessels.items()
            if v['last_snapshot'] and (now - v['last_snapshot'].last_update > self._heartbeat_timeout)
        ]
        if silent:
            events.append(FleetEvent(
                alert_type=FleetAlert.SILENT_FLEET,
                vessels=silent,
                description=f"{len(silent)} vessel(s) silent > {self._heartbeat_timeout}s",
                timestamp=now, severity=0.7,
            ))

        self._events.extend(events)
        return events

    def _detect_cascade(self, vessels: Dict) -> Optional[FleetEvent]:
        """Detect cascading failures using temporal correlation."""
        # Build activity traces
        traces = {}
        for vid, v in vessels.items():
            if len(v['phi_history']) >= 10:
                traces[vid] = list(v['phi_history'])[-32:]

        if len(traces) < 3:
            return None

        conn = TemporalConnectome(threshold=0.5, max_lag=4)
        for vid, trace in traces.items():
            conn.add_room(vid, trace)

        result = conn.analyze()

        # If vessels are highly correlated with lag, it's a cascade
        for pair in result.coupled:
            if pair.lag > 0 and pair.correlation > 0.7:
                # The source vessel (room_a) is affecting the target (room_b)
                return FleetEvent(
                    alert_type=FleetAlert.CASCADE,
                    vessels=[pair.room_a, pair.room_b],
                    description=f"Cascade detected: {pair.room_a} → {pair.room_b} "
                                f"(correlation={pair.correlation:.2f}, lag={pair.lag})",
                    timestamp=time.time(),
                    severity=pair.correlation,
                )

        return None

    def fleet_state(self) -> Dict:
        """Aggregate fleet harmony state."""
        now = time.time()
        vessels = []
        for vid, v in self._vessels.items():
            snap = v['last_snapshot']
            if snap:
                vessels.append({
                    'id': vid,
                    'channel': v['channel'],
                    'phi': snap.phi,
                    'bpm': snap.bpm,
                    'heading': snap.heading,
                    'heading_error': snap.heading_error,
                    'sea_state': snap.sea_state,
                    'harmony': snap.is_in_harmony,
                    'strained': snap.is_strained,
                    'surprised': snap.is_surprised,
                    'age_s': now - snap.last_update,
                    'alarms': snap.alarms,
                })

        global_phi = sum(v['phi'] for v in vessels) / max(len(vessels), 1)

        return {
            'total_vessels': len(self._vessels),
            'active_vessels': sum(1 for v in vessels if v['age_s'] < self._heartbeat_timeout),
            'global_phi': global_phi,
            'vessels_in_harmony': sum(1 for v in vessels if v['harmony']),
            'vessels_strained': sum(1 for v in vessels if v['strained']),
            'vessels_surprised': sum(1 for v in vessels if v['surprised']),
            'vessels': vessels,
            'recent_events': len(self._events),
            'midi_events': self._flux.event_count,
        }

    @property
    def vessels(self) -> List[str]:
        return list(self._vessels.keys())

    @property
    def events(self) -> List[FleetEvent]:
        return list(self._events)

    def clear_events(self) -> None:
        self._events.clear()
