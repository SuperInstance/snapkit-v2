"""
Hypothesis Sandbox — Layer 1 forward simulation and scoring.

Before an agent executes a physical action (adjust throttle, turn rudder),
it runs a micro-simulation testing a functional hypothesis:
    "If I apply value X to Actuator Y, Sensor Z should read W next beat."

The sandbox:
  1. Takes the proposed action (MIDI note)
  2. Runs a forward simulation using the agent's internal model
  3. Scores the prediction against the next tick's actual sensor readings
  4. Returns the óthismos (constraint pressure) value

The score IS óthismos — constraint pressure. Low score = the hypothesis
is functional (the agent's model matches reality). High score = the model
is failing and friction is building.

Zero external dependencies. stdlib only. Python ≥ 3.10.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

from snapkit.spectral import entropy, spectral_summary


@dataclass(frozen=True, slots=True)
class HypothesisResult:
    """Result of a single hypothesis test in the sandbox."""
    action_value: float
    predicted_sensor: float
    confidence: float        # 0-1, based on model fit quality
    sim_score: float         # simulation quality score (lower = better)
    actuation_cost: float    # energy expenditure estimate
    othismos: float          # composite constraint pressure (lower = better)


@dataclass(frozen=True, slots=True)
class SandboxScore:
    """Composite score for a sandbox evaluation cycle."""
    best_hypothesis: HypothesisResult
    all_hypotheses: Tuple[HypothesisResult, ...]
    model_health: float      # 0-1, how well the internal model fits reality
    novelty_recommendation: float  # 0-1, how much novelty injection is needed


class LinearModel:
    """Simple linear forward model for hypothesis testing.

    Learns the relationship: sensor_next = a * action + b * sensor_current + c
    Updates online with each observation.
    """

    __slots__ = ('_a', '_b', '_c', '_lr', '_n', '_error_history')

    def __init__(self, learning_rate: float = 0.01):
        self._a: float = 0.0
        self._b: float = 1.0
        self._c: float = 0.0
        self._lr: float = learning_rate
        self._n: int = 0
        self._error_history: Deque[float] = deque(maxlen=64)

    def predict(self, action: float, sensor_current: float) -> Tuple[float, float]:
        """Predict next sensor value. Returns (prediction, confidence)."""
        prediction = self._a * action + self._b * sensor_current + self._c

        # Confidence based on sample size and recent error
        if self._n < 5:
            conf = 0.1
        else:
            avg_err = sum(self._error_history) / len(self._error_history) if self._error_history else 1.0
            conf = max(0.0, min(1.0, 1.0 / (1.0 + avg_err * 10)))

        return prediction, conf

    def update(self, action: float, sensor_before: float, sensor_after: float) -> float:
        """Update model with observed transition. Returns prediction error."""
        prediction = self._a * action + self._b * sensor_before + self._c
        error = sensor_after - prediction

        # Online gradient descent
        self._a += self._lr * error * action
        self._b += self._lr * error * sensor_before
        self._c += self._lr * error

        abs_error = abs(error)
        self._error_history.append(abs_error)
        self._n += 1

        return abs_error

    @property
    def model_quality(self) -> float:
        """How well the model fits (0-1, higher = better)."""
        if self._n < 5 or not self._error_history:
            return 0.0
        avg_err = sum(self._error_history) / len(self._error_history)
        return max(0.0, min(1.0, 1.0 / (1.0 + avg_err * 10)))

    @property
    def sample_count(self) -> int:
        return self._n


class CorrelationModel:
    """Multi-sensor correlation model for hypothesis testing.

    When a single linear model isn't enough (the boat turns differently
    in different sea states), this model learns correlations between
    multiple sensor inputs and the action's effect.
    """

    __slots__ = ('_weights', '_lr', '_n', '_error_history', '_num_features')

    def __init__(self, num_features: int, learning_rate: float = 0.01):
        self._weights: List[float] = [0.0] * (num_features + 1)  # +1 for action
        self._lr: float = learning_rate
        self._n: int = 0
        self._error_history: Deque[float] = deque(maxlen=64)
        self._num_features: int = num_features

    def predict(self, action: float, features: List[float]) -> Tuple[float, float]:
        """Predict sensor value from action + contextual features."""
        if len(features) != self._num_features:
            raise ValueError(f"expected {self._num_features} features, got {len(features)}")

        inputs = features + [action]
        prediction = sum(w * x for w, x in zip(self._weights, inputs))

        if self._n < 5:
            conf = 0.1
        else:
            avg_err = sum(self._error_history) / len(self._error_history) if self._error_history else 1.0
            conf = max(0.0, min(1.0, 1.0 / (1.0 + avg_err * 10)))

        return prediction, conf

    def update(
        self, action: float, features: List[float], sensor_after: float
    ) -> float:
        """Update model with observed transition."""
        inputs = features + [action]
        prediction = sum(w * x for w, x in zip(self._weights, inputs))
        error = sensor_after - prediction

        for i, x in enumerate(inputs):
            self._weights[i] += self._lr * error * x

        abs_error = abs(error)
        self._error_history.append(abs_error)
        self._n += 1

        return abs_error

    @property
    def model_quality(self) -> float:
        if self._n < 5 or not self._error_history:
            return 0.0
        avg_err = sum(self._error_history) / len(self._error_history)
        return max(0.0, min(1.0, 1.0 / (1.0 + avg_err * 10)))


class HypothesisSandbox:
    """Layer 1: Forward simulation and hypothesis scoring.

    The agent proposes actions. The sandbox simulates their effects using
    the agent's internal model and scores them. The winning hypothesis
    is the one that minimizes óthismos (constraint pressure).

    Usage:
        sandbox = HypothesisSandbox(sensor_name="heading")
        sandbox.set_action_range(0.0, 1.0, step=0.05)  # rudder positions

        # Each cycle:
        sandbox.observe(action_taken=0.3, sensor_before=180.0, sensor_after=182.0)

        # Before acting:
        score = sandbox.evaluate(action=0.4, sensor_current=182.0)
        if score.best_hypothesis.othismos < threshold:
            execute_action(0.4)
    """

    __slots__ = (
        '_sensor_name', '_model', '_action_range',
        '_action_history', '_sensor_history', '_error_history',
        '_actuation_cost_weight', '_sim_cost_weight',
        '_novelty_window', '_last_predictions', '_last_actuals',
    )

    def __init__(
        self,
        sensor_name: str = "sensor",
        model: Optional[LinearModel] = None,
        actuation_cost_weight: float = 0.3,
        sim_cost_weight: float = 0.7,
    ):
        self._sensor_name: str = sensor_name
        self._model: LinearModel = model or LinearModel()
        self._action_range: List[float] = []
        self._action_history: Deque[float] = deque(maxlen=256)
        self._sensor_history: Deque[float] = deque(maxlen=256)
        self._error_history: Deque[float] = deque(maxlen=64)
        self._actuation_cost_weight: float = actuation_cost_weight
        self._sim_cost_weight: float = sim_cost_weight
        self._novelty_window: Deque[float] = deque(maxlen=32)
        self._last_predictions: Deque[float] = deque(maxlen=64)
        self._last_actuals: Deque[float] = deque(maxlen=64)

    def set_action_range(
        self, lo: float, hi: float, step: float = 0.1
    ) -> None:
        """Define the discrete action space to explore."""
        self._action_range = []
        v = lo
        while v <= hi + 1e-9:
            self._action_range.append(round(v, 6))
            v += step

    def observe(
        self,
        action_taken: float,
        sensor_before: float,
        sensor_after: float,
    ) -> float:
        """Feed an observation to update the internal model.

        Returns the prediction error (|predicted - actual|).
        """
        error = self._model.update(action_taken, sensor_before, sensor_after)
        self._action_history.append(action_taken)
        self._sensor_history.append(sensor_after)
        self._error_history.append(error)
        self._last_actuals.append(sensor_after)
        return error

    def evaluate(
        self,
        sensor_current: float,
        candidate_actions: Optional[List[float]] = None,
        target_sensor: Optional[float] = None,
        max_actuation: Optional[float] = None,
    ) -> SandboxScore:
        """Evaluate candidate actions and score them.

        Args:
            sensor_current: Current sensor reading.
            candidate_actions: Actions to test. If None, uses action_range.
            target_sensor: Desired sensor value. If None, minimizes change.
            max_actuation: Maximum allowed |action - last_action|.

        Returns:
            SandboxScore with the best hypothesis and all candidates.
        """
        actions = candidate_actions or self._action_range
        if not actions:
            raise ValueError("no candidate actions — call set_action_range() first")

        last_action = self._action_history[-1] if self._action_history else 0.0

        results: List[HypothesisResult] = []
        for action in actions:
            # Check actuation constraint
            if max_actuation is not None:
                if abs(action - last_action) > max_actuation:
                    continue

            # Forward simulate
            predicted, confidence = self._model.predict(action, sensor_current)

            # Simulation score: how far from target (or how stable)
            if target_sensor is not None:
                sim_score = abs(predicted - target_sensor)
            else:
                # Minimize deviation from current state (stability-seeking)
                sim_score = abs(predicted - sensor_current)

            # Actuation cost: energy of the action relative to current
            if last_action != 0:
                act_cost = abs(action - last_action) / (abs(last_action) + 1e-6)
            else:
                act_cost = abs(action)

            # Composite óthismos: weighted sum of sim quality and actuation cost
            othismos = (
                self._sim_cost_weight * sim_score
                + self._actuation_cost_weight * act_cost
            )

            # Penalty for low confidence
            if confidence < 0.2:
                othismos *= (1.0 + (0.2 - confidence) * 5.0)

            results.append(HypothesisResult(
                action_value=action,
                predicted_sensor=predicted,
                confidence=confidence,
                sim_score=sim_score,
                actuation_cost=act_cost,
                othismos=othismos,
            ))

        if not results:
            # All actions filtered by constraint — return worst possible
            raise ValueError("all candidate actions filtered by max_actuation constraint")

        # Sort by óthismos (lower = better)
        results.sort(key=lambda r: r.othismos)
        best = results[0]

        # Track prediction for novelty assessment
        self._last_predictions.append(best.predicted_sensor)
        self._novelty_window.append(best.othismos)

        # Model health from recent errors
        model_health = self._model.model_quality

        # Novelty recommendation: if recent óthismos is high or rising,
        # the model needs novelty injection
        novelty_rec = self._compute_novelty_recommendation()

        return SandboxScore(
            best_hypothesis=best,
            all_hypotheses=tuple(results),
            model_health=model_health,
            novelty_recommendation=novelty_rec,
        )

    def _compute_novelty_recommendation(self) -> float:
        """Compute how much novelty injection is needed (0-1)."""
        if len(self._novelty_window) < 4:
            return 0.0

        window = list(self._novelty_window)

        # If óthismos is trending up, novelty needed
        half = len(window) // 2
        first_half_avg = sum(window[:half]) / half if half > 0 else 0
        second_half_avg = sum(window[half:]) / (len(window) - half) if len(window) > half else 0

        if first_half_avg > 0:
            trend = (second_half_avg - first_half_avg) / first_half_avg
        else:
            trend = 0.0

        # If errors are high and not improving, recommend novelty
        avg_othismos = sum(window) / len(window)
        base_novelty = min(1.0, avg_othismos / 3.0)
        trend_novelty = max(0.0, min(1.0, trend))

        return min(1.0, base_novelty * 0.5 + trend_novelty * 0.5)

    def prediction_vs_actual_entropy(self) -> float:
        """Entropy of prediction errors — high entropy = model is confused."""
        if len(self._last_predictions) < 4 or len(self._last_actuals) < 4:
            return 0.0

        min_len = min(len(self._last_predictions), len(self._last_actuals))
        errors = [
            abs(p - a)
            for p, a in zip(
                list(self._last_predictions)[-min_len:],
                list(self._last_actuals)[-min_len:],
            )
        ]
        if len(errors) < 4:
            return 0.0

        return entropy(errors, bins=min(5, len(errors) // 2))

    @property
    def model_health(self) -> float:
        """Current internal model quality (0-1)."""
        return self._model.model_quality

    @property
    def sample_count(self) -> int:
        return self._model.sample_count

    @property
    def sensor_name(self) -> str:
        return self._sensor_name

    def state(self) -> Dict:
        """Snapshot of sandbox state."""
        return {
            'sensor': self._sensor_name,
            'model_quality': self.model_health,
            'samples': self.sample_count,
            'action_space_size': len(self._action_range),
            'recent_error_avg': (
                sum(self._error_history) / len(self._error_history)
                if self._error_history else 0.0
            ),
            'prediction_entropy': self.prediction_vs_actual_entropy(),
        }
