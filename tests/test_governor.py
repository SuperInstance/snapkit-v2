"""Tests for the Harmony Governor."""

import pytest
from snapkit.governor import (
    HarmonyGovernor, ChannelState, FrictionAlarm,
    FrictionLevel, HarmonyGovernor as HG,
)
from snapkit.midi import TempoMap


class TestChannelState:
    def test_initial_state(self):
        cs = ChannelState("helm", channel=0, deadband=2.0)
        assert cs.phi == 0.0
        assert cs.is_in_harmony
        assert not cs.is_strained
        assert not cs.is_surprised

    def test_record_builds_history(self):
        cs = ChannelState("helm", channel=0, window_size=16)
        for i in range(10):
            cs.record(prediction=0.5, actual=0.5, latency_ms=10, tick=i)
        assert len(cs.actuals) == 10
        assert cs.phi >= 0.0

    def test_prediction_error_increases_phi(self):
        cs = ChannelState("helm", channel=0, window_size=32, deadband=0.5)
        # Perfect predictions = low phi
        for i in range(20):
            cs.record(prediction=1.0, actual=1.0, latency_ms=5, tick=i)
        phi_harmony = cs.phi

        # Bad predictions = high phi
        cs2 = ChannelState("bad", channel=1, window_size=32, deadband=0.5)
        for i in range(20):
            cs2.record(prediction=1.0, actual=0.0, latency_ms=500, tick=i)
        phi_surprise = cs2.phi

        assert phi_surprise > phi_harmony


class TestHarmonyGovernor:
    def test_register_channel(self):
        gov = HarmonyGovernor()
        gov.register_channel("helm", channel=0)
        assert "helm" in gov.channels

    def test_duplicate_channel_raises(self):
        gov = HarmonyGovernor()
        gov.register_channel("helm", channel=0)
        with pytest.raises(ValueError):
            gov.register_channel("helm", channel=1)

    def test_tick_advances(self):
        gov = HarmonyGovernor()
        gov.tick(10)
        assert gov.tick_count == 10
        gov.tick(11)
        assert gov.tick_count == 11

    def test_record_observation(self):
        gov = HarmonyGovernor()
        gov.register_channel("helm", channel=0)
        gov.tick(0)
        phi = gov.record_observation("helm", prediction=0.5, actual=0.55, latency_ms=10)
        assert phi >= 0.0

    def test_alarm_on_sustained_surprise(self):
        alarms = []
        gov = HarmonyGovernor(
            sustained_threshold=3,
            on_alarm=lambda a: alarms.append(a),
        )
        gov.register_channel("failing", channel=0, deadband=0.1, window_size=8)
        for i in range(10):
            gov.tick(i)
            gov.record_observation(
                "failing", prediction=1.0, actual=0.0, latency_ms=2000,
            )
        assert len(gov.unacknowledged_alarms) > 0
        assert any(a.level == FrictionLevel.SURPRISE for a in gov.alarms)

    def test_harmony_no_alarms(self):
        gov = HarmonyGovernor(sustained_threshold=2)
        gov.register_channel("stable", channel=0, deadband=5.0)
        for i in range(10):
            gov.tick(i)
            gov.record_observation(
                "stable", prediction=1.0, actual=1.01, latency_ms=5,
            )
        assert len(gov.unacknowledged_alarms) == 0

    def test_clear_alarms(self):
        gov = HarmonyGovernor()
        gov._alarms.append(
            FrictionAlarm(0, "test", FrictionLevel.SURPRISE, 3.0, 1.0, 0.4, 100, 0, 0)
        )
        assert len(gov.alarms) == 1
        gov.clear_alarms()
        assert len(gov.alarms) == 0

    def test_global_phi(self):
        gov = HarmonyGovernor()
        gov.register_channel("a", channel=0)
        gov.register_channel("b", channel=1)
        gov.tick(0)
        gov.record_observation("a", prediction=0.5, actual=0.5, latency_ms=5)
        gov.record_observation("b", prediction=0.5, actual=0.5, latency_ms=5)
        assert gov.global_phi() >= 0.0

    def test_system_state(self):
        gov = HarmonyGovernor()
        gov.register_channel("helm", channel=0)
        gov.tick(0)
        gov.record_observation("helm", prediction=0.5, actual=0.5, latency_ms=10)
        state = gov.system_state()
        assert state["tick"] == 0
        assert "helm" in state["channels"]
        assert "global_phi" in state
