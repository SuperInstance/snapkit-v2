"""Tests for the MIDI I/O Bridge."""

import pytest
from snapkit.governor import HarmonyGovernor
from snapkit.midi_io import (
    MIDIBridge, SensorMapper, TempoDeriver,
    SensorReading, AgentAction,
    SENSOR_CC_MAP, ACTION_NOTE_MAP,
)


class TestSensorMapper:
    def test_to_midi_basic(self):
        assert SensorMapper.to_midi(0, 0, 100) == 0
        assert SensorMapper.to_midi(100, 0, 100) == 127
        assert SensorMapper.to_midi(50, 0, 100) == 63

    def test_to_midi_clamps(self):
        assert SensorMapper.to_midi(-10, 0, 100) == 0
        assert SensorMapper.to_midi(200, 0, 100) == 127

    def test_from_midi_roundtrip(self):
        for val in [0, 32, 64, 96, 127]:
            physical = SensorMapper.from_midi(val, 0, 360)
            back = SensorMapper.to_midi(physical, 0, 360)
            assert abs(back - val) <= 1

    def test_angle_to_midi(self):
        assert SensorMapper.angle_to_midi(0) == 0
        assert SensorMapper.angle_to_midi(180) == 63
        assert SensorMapper.angle_to_midi(360) == 0  # wraps

    def test_midi_to_angle(self):
        assert abs(SensorMapper.midi_to_angle(0) - 0) < 0.1
        assert abs(SensorMapper.midi_to_angle(63) - 179) < 2
        assert abs(SensorMapper.midi_to_angle(127) - 360) < 0.1


class TestTempoDeriver:
    def test_initial_state(self):
        td = TempoDeriver()
        assert td.bpm > 0
        assert td.period > 0
        assert td.stability == 0.0

    def test_feed_accumulates(self):
        td = TempoDeriver()
        for i in range(50):
            td.feed(float(i % 10))
        assert td.state()["samples"] == 50

    def test_periodic_signal_sets_tempo(self):
        td = TempoDeriver(window_size=256)
        # Simulate a 4-second roll period at 10 Hz sampling
        for i in range(200):
            import math
            roll = 5.0 * math.sin(2 * math.pi * i / 40.0)
            td.feed(roll)
        # Should detect some periodicity
        assert td.stability > 0.0

    def test_state_snapshot(self):
        td = TempoDeriver()
        for i in range(40):
            td.feed(float(i % 5))
        state = td.state()
        assert "bpm" in state
        assert "period_seconds" in state
        assert "stability" in state


class TestMIDIBridge:
    def test_register_sensor(self):
        bridge = MIDIBridge()
        bridge.register_sensor("heading", lo=0, hi=360, source="nmea2000")
        assert "heading" in bridge.sensors

    def test_feed_sensor(self):
        bridge = MIDIBridge()
        bridge.register_sensor("heading", lo=0, hi=360, source="nmea2000")
        reading = bridge.feed_sensor("heading", 180.0)
        assert isinstance(reading, SensorReading)
        assert reading.raw_value == 180.0
        assert reading.midi_value == 63

    def test_feed_sensor_unregistered_raises(self):
        bridge = MIDIBridge()
        with pytest.raises(KeyError):
            bridge.feed_sensor("nonexistent", 1.0)

    def test_feed_roll_returns_bpm(self):
        bridge = MIDIBridge()
        bpm = bridge.feed_roll(5.0)
        assert bpm > 0

    def test_agent_action(self):
        gov = HarmonyGovernor()
        gov.register_channel("helm", channel=0)
        bridge = MIDIBridge(governor=gov)
        action = bridge.agent_action(
            "helm", "rudder_left", velocity=100, confidence=0.9,
        )
        assert isinstance(action, AgentAction)
        assert action.note_number == ACTION_NOTE_MAP["rudder_left"]
        assert action.velocity > 0

    def test_with_governor(self):
        gov = HarmonyGovernor()
        gov.register_channel("helm", channel=0)
        bridge = MIDIBridge(governor=gov)
        bridge.register_sensor("heading", lo=0, hi=360)

        gov.tick(0)
        bridge.feed_sensor("heading", 180.0)

        assert gov._flux.event_count > 0

    def test_playback_events(self):
        gov = HarmonyGovernor()
        gov.register_channel("helm", channel=0)
        bridge = MIDIBridge(governor=gov)
        gov.tick(0)
        bridge.agent_action("helm", "rudder_left", velocity=80)
        events = bridge.playback_events()
        assert len(events) > 0

    def test_state_snapshot(self):
        bridge = MIDIBridge()
        bridge.register_sensor("depth", lo=0, hi=200, source="nmea2000")
        bridge.feed_roll(3.0)
        state = bridge.state()
        assert "sensors" in state
        assert "depth" in state["sensors"]
        assert "tempo" in state

    def test_sensor_cc_map_has_standard_entries(self):
        assert "rudder_angle" in SENSOR_CC_MAP
        assert "heading" in SENSOR_CC_MAP
        assert "roll_angle" in SENSOR_CC_MAP

    def test_action_note_map_has_standard_entries(self):
        assert "rudder_left" in ACTION_NOTE_MAP
        assert "mayday" in ACTION_NOTE_MAP
        assert "engine_stop" in ACTION_NOTE_MAP
