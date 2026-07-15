"""Tests for the Hypothesis Sandbox."""

import pytest
from snapkit.sandbox import (
    HypothesisSandbox, LinearModel, CorrelationModel,
    HypothesisResult, SandboxScore,
)


class TestLinearModel:
    def test_initial_predict(self):
        m = LinearModel()
        pred, conf = m.predict(action=0.5, sensor_current=1.0)
        assert isinstance(pred, float)
        assert 0.0 <= conf <= 1.0

    def test_update_improves_fit(self):
        m = LinearModel(learning_rate=0.1)
        # True relationship: sensor_next = 2 * action + sensor_current
        errors = []
        for i in range(50):
            action = 0.3 + (i % 5) * 0.1
            before = 1.0 + i * 0.01
            after = 2.0 * action + before
            err = m.update(action, before, after)
            errors.append(err)
        # Later errors should be smaller than early errors
        avg_first = sum(errors[:5]) / 5
        avg_last = sum(errors[-5:]) / 5
        assert avg_last < avg_first

    def test_model_quality_improves(self):
        m = LinearModel(learning_rate=0.1)
        q1 = m.model_quality
        for i in range(20):
            m.update(0.5, 1.0, 1.5)
        q2 = m.model_quality
        assert q2 >= q1


class TestCorrelationModel:
    def test_predict_with_features(self):
        m = CorrelationModel(num_features=2)
        pred, conf = m.predict(action=0.5, features=[1.0, 0.3])
        assert isinstance(pred, float)

    def test_wrong_feature_count_raises(self):
        m = CorrelationModel(num_features=2)
        with pytest.raises(ValueError):
            m.predict(action=0.5, features=[1.0])


class TestHypothesisSandbox:
    def test_set_action_range(self):
        sb = HypothesisSandbox("heading")
        sb.set_action_range(0.0, 1.0, step=0.1)
        score = sb.evaluate(sensor_current=0.5)
        assert len(score.all_hypotheses) == 11

    def test_observe_updates_model(self):
        sb = HypothesisSandbox("heading")
        sb.set_action_range(0.0, 1.0, step=0.1)
        for i in range(10):
            sb.observe(action_taken=0.5, sensor_before=1.0, sensor_after=1.5)
        assert sb.sample_count == 10
        assert sb.model_health > 0.0

    def test_evaluate_returns_best(self):
        sb = HypothesisSandbox("heading")
        sb.set_action_range(0.0, 1.0, step=0.1)
        for i in range(20):
            sb.observe(action_taken=0.5, sensor_before=1.0, sensor_after=1.5)

        score = sb.evaluate(sensor_current=1.0)
        assert isinstance(score, SandboxScore)
        assert isinstance(score.best_hypothesis, HypothesisResult)
        assert score.best_hypothesis.othismos >= 0.0
        assert len(score.all_hypotheses) > 0

    def test_othismos_favors_stability(self):
        """When no target, óthismos should favor minimal actuation."""
        # With actuation cost enabled, the zero action should win
        sb = HypothesisSandbox(
            "heading", sim_cost_weight=0.3, actuation_cost_weight=0.7,
        )
        sb.set_action_range(-1.0, 1.0, step=0.1)
        for i in range(20):
            sb.observe(action_taken=0.0, sensor_before=1.0, sensor_after=1.0)

        score = sb.evaluate(sensor_current=1.0)
        # With actuation cost weight > 0, low-action hypotheses should win
        assert abs(score.best_hypothesis.action_value) <= 0.3

    def test_target_seeking(self):
        """With a target, óthismos should favor reaching it."""
        sb = HypothesisSandbox(
            "heading", sim_cost_weight=1.0, actuation_cost_weight=0.0,
        )
        sb.set_action_range(-1.0, 1.0, step=0.1)

        # Train: action 1.0 → sensor increases by 1.0
        for i in range(30):
            sb.observe(action_taken=1.0, sensor_before=0.0, sensor_after=1.0)

        score = sb.evaluate(
            sensor_current=0.0,
            target_sensor=1.0,
        )
        # The model should have learned that action → sensor change.
        # Best hypothesis should be high action (closest to target)
        assert score.best_hypothesis.action_value >= 0.5

    def test_max_actuation_constraint(self):
        sb = HypothesisSandbox("heading")
        sb.set_action_range(0.0, 1.0, step=0.1)
        for i in range(10):
            sb.observe(action_taken=0.0, sensor_before=1.0, sensor_after=1.0)

        score = sb.evaluate(
            sensor_current=1.0,
            max_actuation=0.15,
        )
        # All hypotheses should be within 0.15 of last action (0.0)
        for h in score.all_hypotheses:
            assert abs(h.action_value) <= 0.15 + 1e-6

    def test_novelty_recommendation(self):
        sb = HypothesisSandbox("heading")
        sb.set_action_range(0.0, 1.0, step=0.1)

        # Consistently bad predictions → high novelty recommendation
        for i in range(20):
            sb.observe(action_taken=0.5, sensor_before=float(i), sensor_after=float(i + 100))

        score = sb.evaluate(sensor_current=100.0)
        assert score.novelty_recommendation >= 0.0

    def test_prediction_entropy(self):
        sb = HypothesisSandbox("heading")
        sb.set_action_range(0.0, 1.0, step=0.1)
        for i in range(20):
            sb.observe(action_taken=0.5, sensor_before=1.0, sensor_after=1.5)

        ent = sb.prediction_vs_actual_entropy()
        assert ent >= 0.0

    def test_state_snapshot(self):
        sb = HypothesisSandbox("heading")
        sb.set_action_range(0.0, 1.0, step=0.1)
        for i in range(5):
            sb.observe(action_taken=0.5, sensor_before=1.0, sensor_after=1.5)
        state = sb.state()
        assert state["sensor"] == "heading"
        assert state["samples"] == 5

    def test_empty_action_range_raises(self):
        sb = HypothesisSandbox("heading")
        with pytest.raises(ValueError):
            sb.evaluate(sensor_current=1.0)
