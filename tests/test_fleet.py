"""Tests for the Fleet Coordinator."""

import pytest
import time
from snapkit.fleet import (
    FleetCoordinator, VesselSnapshot, FleetEvent, FleetAlert,
)


def make_snapshot(
    vessel_id="v1", channel=0, phi=0.2, heading=180.0,
    harmony=True, strained=False, surprised=False,
    alarms=0, age=0.0,
):
    return VesselSnapshot(
        vessel_id=vessel_id, channel=channel,
        phi=phi, bpm=24.0, heading=heading,
        heading_error=abs(heading - 180),
        sea_state="calm",
        is_in_harmony=harmony, is_strained=strained, is_surprised=surprised,
        last_update=time.time() - age, alarms=alarms,
    )


class TestFleetCoordinator:
    def test_register_vessel(self):
        fleet = FleetCoordinator()
        fleet.register_vessel("v1", channel=0)
        assert "v1" in fleet.vessels

    def test_duplicate_vessel_raises(self):
        fleet = FleetCoordinator()
        fleet.register_vessel("v1", channel=0)
        with pytest.raises(ValueError):
            fleet.register_vessel("v1", channel=1)

    def test_update_vessel(self):
        fleet = FleetCoordinator()
        fleet.register_vessel("v1", channel=0)
        fleet.update_vessel(make_snapshot("v1", phi=0.1))
        state = fleet.fleet_state()
        assert state["total_vessels"] == 1
        assert state["active_vessels"] == 1

    def test_single_vessel_alert(self):
        fleet = FleetCoordinator()
        fleet.register_vessel("v1", channel=0)
        fleet.register_vessel("v2", channel=1)
        fleet.update_vessel(make_snapshot("v1", harmony=True))
        fleet.update_vessel(make_snapshot(
            "v2", surprised=True, harmony=False,
        ))
        events = fleet.detect_events()
        assert any(e.alert_type == FleetAlert.SINGLE_VESSEL for e in events)

    def test_coordinated_failure(self):
        fleet = FleetCoordinator()
        fleet.register_vessel("v1", channel=0)
        fleet.register_vessel("v2", channel=1)
        fleet.update_vessel(make_snapshot(
            "v1", surprised=True, harmony=False,
        ))
        fleet.update_vessel(make_snapshot(
            "v2", surprised=True, harmony=False,
        ))
        events = fleet.detect_events()
        assert any(e.alert_type == FleetAlert.COORDINATED for e in events)

    def test_drift_detection(self):
        fleet = FleetCoordinator()
        fleet.register_vessel("v1", channel=0)
        fleet.register_vessel("v2", channel=1)
        fleet.update_vessel(make_snapshot("v1", heading=170))
        fleet.update_vessel(make_snapshot("v2", heading=230))
        events = fleet.detect_events()
        assert any(e.alert_type == FleetAlert.DRIFT for e in events)

    def test_silent_fleet(self):
        fleet = FleetCoordinator(heartbeat_timeout=0.1)
        fleet.register_vessel("v1", channel=0)
        fleet.update_vessel(make_snapshot("v1", age=5.0))
        events = fleet.detect_events()
        assert any(e.alert_type == FleetAlert.SILENT_FLEET for e in events)

    def test_fleet_state(self):
        fleet = FleetCoordinator()
        fleet.register_vessel("v1", channel=0)
        fleet.register_vessel("v2", channel=1)
        fleet.update_vessel(make_snapshot("v1", phi=0.1))
        fleet.update_vessel(make_snapshot("v2", phi=0.5))
        state = fleet.fleet_state()
        assert state["total_vessels"] == 2
        assert state["vessels_in_harmony"] == 2
        assert "global_phi" in state

    def test_clear_events(self):
        fleet = FleetCoordinator()
        fleet._events.append(
            FleetEvent(FleetAlert.DRIFT, ["v1"], "test", 0, 0.5)
        )
        assert len(fleet.events) == 1
        fleet.clear_events()
        assert len(fleet.events) == 0

    def test_unknown_vessel_ignored(self):
        fleet = FleetCoordinator()
        fleet.register_vessel("v1", channel=0)
        fleet.update_vessel(make_snapshot("unknown_vessel"))
        state = fleet.fleet_state()
        assert state["total_vessels"] == 1
