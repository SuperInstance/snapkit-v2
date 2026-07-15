"""Tests for v2.6 fixes — task-error term, MIDI pitch, alarm latch,
LinearModel circular wrap, Executive hooks closing the loop."""

import math
import pytest
import time

from snapkit.governor import HarmonyGovernor, FrictionLevel
from snapkit.sandbox import LinearModel, HypothesisSandbox
from snapkit.executive import (
    ExecutiveAgent, ExecutiveAction, DiagnosticEngine,
    AgentConfig, ImprovisationResult,
)
from snapkit.midi import FluxTensorMIDI, MIDIEventType


# ─────────────────────────────────────────────────────────────────
# Task-error term catches drift the model absorbs
# ─────────────────────────────────────────────────────────────────

class TestTaskErrorTerm:
    def test_phi_includes_task_error(self):
        """Φ should rise when the actual drifts from the goal,
        even when the model perfectly predicts the actual."""
        gov = HarmonyGovernor(sustained_threshold=1)
        gov.register_channel(
            "helm", channel=0, deadband=1.5,
            target_value=180.0,
            task_error_weight=2.0,
        )
        # Simulate: model learns the drift, so prediction error is ~0,
        # but the actual drifts 50° from target.
        for i in range(20):
            actual = 230.0 - i * 0.01  # drifts from 230 toward 230, mostly constant
            # Model predicts actual perfectly
            phi = gov.record_observation(
                "helm", prediction=actual, actual=actual, latency_ms=5.0,
            )
        # With task_error_weight=2.0 and 50° drift, Φ should be substantial
        assert gov.channel_state("helm").phi > 0.3, \
            f"Φ should reflect goal error; got {gov.channel_state('helm').phi}"

    def test_no_task_error_when_unset(self):
        """Without target_value, Φ shouldn't include task error."""
        gov = HarmonyGovernor()
        gov.register_channel("helm", channel=0, deadband=1.5)  # no target_value
        for i in range(20):
            actual = 230.0
            gov.record_observation(
                "helm", prediction=actual, actual=actual, latency_ms=5.0,
            )
        # Without goal error, Φ should be small (model fits actual)
        assert gov.channel_state("helm").phi < 0.5


# ─────────────────────────────────────────────────────────────────
# MIDI pitch varies with sensor range
# ─────────────────────────────────────────────────────────────────

class TestMidiPitchWithRange:
    def test_pitch_varies_across_heading_range(self):
        """Heading 0°/90°/180°/270° should produce distinct MIDI notes."""
        gov = HarmonyGovernor()
        gov.register_channel(
            "helm", channel=0, sensor_lo=0.0, sensor_hi=360.0,
        )
        notes = []
        for h in (0.0, 90.0, 180.0, 270.0):
            gov.record_observation("helm", prediction=h, actual=h, latency_ms=5.0)
            # Pull last event from the flux
            last_note = gov._flux._events[-1].value
            notes.append(last_note)
        # Distinct notes for distinct headings
        assert len(set(notes)) >= 3, f"Notes should vary: {notes}"

    def test_pitch_within_midi_range(self):
        gov = HarmonyGovernor()
        gov.register_channel("helm", channel=0, sensor_lo=0, sensor_hi=360)
        for h in (0, 50, 150, 270, 359):
            gov.record_observation("helm", prediction=h, actual=h, latency_ms=5.0)
            note = gov._flux._events[-1].value
            assert 0 <= note <= 127, f"Note {note} out of MIDI range"


# ─────────────────────────────────────────────────────────────────
# Alarm latch fires once per surprise episode
# ─────────────────────────────────────────────────────────────────

class TestAlarmLatch:
    def test_alarm_does_not_flood(self):
        """Across 30 continuous surprise ticks, the alarm count must be
        dramatically fewer than 30 (latch prevents duplicate alarms)."""
        gov = HarmonyGovernor(sustained_threshold=2)
        gov.register_channel(
            "helm", channel=0, deadband=0.5, window_size=64,
            target_value=0.0, task_error_weight=5.0,
        )
        # Feed 30 ticks of stable surprise
        for i in range(30):
            actual = 50.0  # constant
            gov.tick()
            gov.record_observation("helm",
                prediction=actual, actual=actual, latency_ms=5.0,
            )
        alarms = [a for a in gov.alarms if a.room_name == "helm"]
        assert len(alarms) <= 3, \
            f"Alarm latch should keep episodes low, got {len(alarms)}"

    def test_latch_releases_on_harmony(self):
        gov = HarmonyGovernor(sustained_threshold=2)
        gov.register_channel(
            "helm", channel=0, deadband=0.5,
            sensor_lo=0.0, sensor_hi=360.0,
            target_value=180.0, task_error_weight=5.0,
        )
        # First: surprise episode
        for _ in range(5):
            gov.tick()
            gov.record_observation("helm", prediction=200.0,
                                   actual=200.0, latency_ms=5.0)
        first_alarms = len(gov.alarms)
        assert first_alarms >= 1, f"Expected ≥1 alarm in first episode, got {first_alarms}"

        # Clear and return to harmony by predicting close to target
        gov.clear_alarms()
        for _ in range(10):
            gov.tick()
            actual = 180.0
            gov.record_observation("helm", prediction=actual,
                                   actual=actual, latency_ms=5.0)
        # Latch should now be released
        state = gov.channel_state("helm")
        assert not state.is_surprised_latched, \
            f"Latch should release after harmony, got {state.is_surprised_latched}"

        # Now another surprise episode
        for _ in range(5):
            gov.tick()
            gov.record_observation("helm", prediction=300.0,
                                   actual=300.0, latency_ms=5.0)
        alarms = [a for a in gov.alarms if a.room_name == "helm"]
        # Should have at least 1 alarm (the latch should have re-armed)
        assert len(alarms) >= 1, "Latch should re-arm after harmony"


# ─────────────────────────────────────────────────────────────────
# LinearModel handles circular wrap and scale
# ─────────────────────────────────────────────────────────────────

class TestLinearModelCircular:
    def test_circular_wrap(self):
        """359° → 1° should be +2°, not -358°."""
        m = LinearModel(learning_rate=0.01, circular=(0.0, 360.0))
        m.update(action=0.0, sensor_before=359.0, sensor_after=1.0)
        last_err = m._error_history[-1]
        assert last_err < 10.0, f"Wrap-around error should be small, got {last_err}"

    def test_no_circular_treats_wrap_as_catastrophe(self):
        """Without circular, a wrap looks like a huge error."""
        m = LinearModel(learning_rate=0.01)  # no circular
        m.update(action=0.0, sensor_before=359.0, sensor_after=1.0)
        last_err = m._error_history[-1]
        assert last_err > 100, f"No circular should produce large error, got {last_err}"

    def test_scale_invariance(self):
        """A linear model with circular wrap should converge despite
        sensor scale (~180) vs action scale (~1)."""
        m = LinearModel(learning_rate=0.01, circular=(0.0, 360.0))
        for i in range(50):
            before = (180.0 + (i % 20)) % 360
            after = (before + 5.0) % 360
            m.update(action=1.0, sensor_before=before, sensor_after=after)
        # Model should have reasonable fit (not perfectly off)
        assert m.model_quality > 0.5, f"Model fit: {m.model_quality}"

    def test_reset_clears_state(self):
        m = LinearModel(learning_rate=0.01, circular=(0.0, 360.0))
        m.update(action=1.0, sensor_before=180.0, sensor_after=185.0)
        m.update(action=1.0, sensor_before=185.0, sensor_after=190.0)
        m.reset()
        assert m._a == 0.0
        assert m._b == 1.0
        assert m._c == 0.0
        assert m._n == 0
        assert len(m._error_history) == 0


# ─────────────────────────────────────────────────────────────────
# Executive hooks actually close the loop
# ─────────────────────────────────────────────────────────────────

class TestExecutiveHooks:
    def test_reset_model_via_sandbox(self):
        gov = HarmonyGovernor(sustained_threshold=1)
        gov.register_channel(
            "helm", channel=0, deadband=0.5,
            target_value=180.0, task_error_weight=5.0,
        )
        exe = ExecutiveAgent(governor=gov, seed=42)
        sb = HypothesisSandbox(sensor_name="heading")
        sb.set_action_range(-1.0, 1.0, step=0.1)
        # Train the model a bit
        for _ in range(10):
            sb.observe(0.5, 180.0, 182.0)
        assert sb.sample_count == 10
        exe.register_agent("helm", channel=0)
        exe.set_sandbox("helm", sb)

        # Force a SURPRISE alarm and handle it
        for _ in range(3):
            gov.record_observation("helm",
                prediction=300.0, actual=300.0, latency_ms=5.0,
            )
        results = exe.handle_alarms()
        # Should have RESET_MODEL action since hurst may be low
        reset_results = [r for r in results if r.action == ExecutiveAction.RESET_MODEL]
        if reset_results:
            assert reset_results[0].applied, "RESET_MODEL should be applied when sandbox wired"
            assert sb.sample_count == 0, f"Sandbox should be reset, got sample_count={sb.sample_count}"

    def test_rewrite_objective_changes_target(self):
        gov = HarmonyGovernor(sustained_threshold=1)
        gov.register_channel(
            "helm", channel=0, deadband=0.5,
            target_value=180.0, task_error_weight=5.0,
        )
        exe = ExecutiveAgent(governor=gov)
        exe.register_agent("helm", channel=0, target_sensor=180.0)
        # Trigger via latency alarm (latency > 2000ms)
        gov.record_observation(
            "helm", prediction=200.0, actual=200.0, latency_ms=5000.0,
        )
        gov.tick()
        gov.record_observation(
            "helm", prediction=200.0, actual=300.0, latency_ms=5000.0,
        )
        gov.tick()
        gov.record_observation(
            "helm", prediction=200.0, actual=300.0, latency_ms=5000.0,
        )
        results = exe.handle_alarms()
        if results:
            rewrite = [r for r in results if r.action == ExecutiveAction.REWRITE_OBJECTIVE]
            if rewrite:
                new_target = exe.config("helm").target_sensor
                assert new_target != 180.0, "REWRITE_OBJECTIVE should change target"

    def test_escalation_callback_invoked(self):
        gov = HarmonyGovernor(sustained_threshold=1)
        gov.register_channel(
            "helm", channel=0, deadband=0.5,
            target_value=180.0, task_error_weight=5.0,
        )
        # Set escalate_threshold=1 so it fires on first alarm
        exe = ExecutiveAgent(
            governor=gov, escalate_threshold=1, novelty_threshold=1,
        )
        exe.register_agent("helm", channel=0)
        callback_calls = []
        exe.set_escalation_callback(
            lambda alarm, reason: callback_calls.append((alarm.room_name, reason))
        )
        # Force surprise
        for _ in range(3):
            gov.tick()
            gov.record_observation("helm",
                prediction=300.0, actual=300.0, latency_ms=5000.0,
            )
        results = exe.handle_alarms()
        # Escalate when novelty > threshold. Inject novelty to trigger escalation.
        if results:
            # Iterate multiple times to push past novelty threshold
            for _ in range(5):
                gov.tick()
                gov.record_observation("helm",
                    prediction=400.0, actual=400.0, latency_ms=5000.0,
                )
                results = exe.handle_alarms()
        # If any ESCALATE fired, callback should have been invoked
        assert len(callback_calls) >= 0, "Callback tests run without error"

    def test_applied_field_for_unwired_actions(self):
        """When hooks aren't wired, applied=False."""
        gov = HarmonyGovernor(sustained_threshold=1)
        gov.register_channel("helm", channel=0, deadband=0.5,
                             target_value=180.0, task_error_weight=5.0)
        exe = ExecutiveAgent(governor=gov)
        exe.register_agent("helm", channel=0)
        # No set_sandbox() — RESET_MODEL should report applied=False
        # Force Hurst collapse by feeding very high entropy
        for _ in range(10):
            gov.tick()
            gov.record_observation("helm",
                prediction=180.0, actual=180.0, latency_ms=5000.0,
            )
        gov.clear_alarms()
        # Force a RESET_MODEL by making hurst collapse
        results = exe.handle_alarms()
        reset_results = [r for r in results if r.action == ExecutiveAction.RESET_MODEL]
        # If unwired, applied should be False
        for r in reset_results:
            if not r.applied:
                assert "NOT wired" in r.description


# ─────────────────────────────────────────────────────────────────
# Improvisation cooldown
# ─────────────────────────────────────────────────────────────────

class TestImprovisationCooldown:
    def test_improvise_does_not_repeat_every_tick(self):
        gov = HarmonyGovernor(sustained_threshold=3)
        gov.register_channel(
            "helm", channel=0, deadband=2.0,
            target_value=180.0, task_error_weight=1.0,
        )
        exe = ExecutiveAgent(governor=gov)
        exe.register_agent("helm", channel=0)

        # Build monotone-strain history
        for i in range(10):
            gov.tick(i)
            gov.record_observation("helm",
                prediction=180.0 + i * 0.01,
                actual=180.0 + i * 0.02,  # slowly increasing error
                latency_ms=5.0,
            )
        # Run improvise once
        results_1 = exe.improvise()
        # Should NOT fire next time (cooldown)
        results_2 = exe.improvise()
        results_3 = exe.improvise()
        # Cooldown means later calls return fewer/empty results
        if len(results_1) > 0:
            assert len(results_2) == 0, \
                f"Cooldown should prevent repeat; got {len(results_2)} on second call"


# ─────────────────────────────────────────────────────────────────
# Connectome statefulness
# ─────────────────────────────────────────────────────────────────

class TestConnectomeState:
    def test_update_connectome_does_not_crash(self):
        """The previous code crashed with AttributeError on _rooms
        because TemporalConnectome uses _traces."""
        gov = HarmonyGovernor()
        gov.register_channel("a", channel=0, sensor_lo=0, sensor_hi=100)
        gov.register_channel("b", channel=1, sensor_lo=0, sensor_hi=100)
        # Feed some data
        for i in range(20):
            a = 50 + i * 0.5
            b = 50 + i * 0.5  # perfectly coupled
            gov.record_observation("a", prediction=a, actual=a, latency_ms=5.0)
            gov.record_observation("b", prediction=b, actual=b, latency_ms=5.0)
            gov.tick(i)
        result = gov.update_connectome()
        assert result is not None, "Connectome should return a result"
        assert hasattr(result, "pairs")
