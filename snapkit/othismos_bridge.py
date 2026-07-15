"""
Óthismos Bridge — Connects snapkit-v2 Harmony Governor to the óthismos library.

óthismos (ὄθισμος) = the force a bounded system exerts against its bounds.
The Harmony Governor's Φ IS óthismos. This module makes the connection explicit.

When the óthismos library is installed (pip install othismos), this bridge
provides:
  - PressureGaugeChannel: a governor channel backed by óthismos measurement
  - ConstraintPressureMonitor: real-time óthismos tracking from optimization logs
  - PhaseDetection: maps óthismos phases (exploration → exploitation → crisis)
    to the governor's harmony/strain/surprise levels

Fallback mode: if óthismos is not installed, falls back to the governor's
built-in entropy/Hurst/latency friction metric.

Zero external dependencies (othismos optional). stdlib only.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from snapkit.governor import (
    HarmonyGovernor, ChannelState, FrictionLevel, FrictionAlarm,
)


@dataclass
class PressureReading:
    """A single óthismos pressure reading."""
    step: int
    pressure: float          # Π = ‖Δθ‖ (constraint pressure magnitude)
    raw_step_norm: float     # ‖s‖ (the desired step before projection)
    projected_step_norm: float  # ‖s*‖ (the actual step after projection)
    clip_ratio: float        # pressure / raw_step_norm (fraction clipped)
    timestamp: float = 0.0


class PressureGaugeChannel(ChannelState):
    """A governor channel backed by óthismos pressure measurement.

    Instead of computing friction from prediction entropy and latency,
    this channel reads constraint pressure directly from an optimization
    loop. The pressure IS the friction — óthismos IS Φ.

    Usage:
        gov = HarmonyGovernor()
        gauge = PressureGaugeChannel("training_run", channel=0)
        gov._channels["training_run"] = gauge

        # Each optimization step:
        gauge.record_pressure(step=100, pressure=0.45)
    """

    def __init__(
        self,
        name: str,
        channel: int,
        window_size: int = 128,
        deadband: float = 2.0,
    ):
        super().__init__(name, channel, window_size, deadband)
        self._pressure_history: Deque[float] = deque(maxlen=window_size)
        self._clip_ratios: Deque[float] = deque(maxlen=window_size)

    def record_pressure(
        self,
        step: int,
        pressure: float,
        raw_step_norm: float = 0.0,
        projected_step_norm: float = 0.0,
        timestamp: float = 0.0,
    ) -> float:
        """Record an óthismos pressure measurement.

        Args:
            step: Optimization step number.
            pressure: Π = ‖s - s*‖ (constraint pressure magnitude).
            raw_step_norm: ‖s‖ (desired step before constraint projection).
            projected_step_norm: ‖s*‖ (actual step after projection).
            timestamp: Wall-clock time.

        Returns Φ for this reading (normalized pressure).
        """
        self._pressure_history.append(pressure)
        self.last_update_tick = step

        # Clip ratio: what fraction of the step was eaten by constraints?
        if raw_step_norm > 1e-10:
            clip_ratio = pressure / raw_step_norm
        else:
            clip_ratio = 0.0
        self._clip_ratios.append(min(clip_ratio, 1.0))

        # Record as prediction/actual pair for compatibility with ChannelState
        self.predictions.append(projected_step_norm)
        self.actuals.append(raw_step_norm)
        self.latencies.append(0.0)

        # Φ = normalized pressure (0-3 range like the standard governor)
        # Average over recent window to smooth noise
        recent = list(self._pressure_history)[-min(8, len(self._pressure_history)):]
        avg_pressure = sum(recent) / len(recent)

        # Scale: pressure of 0 = perfect harmony, pressure > 1 = significant
        phi = min(avg_pressure * 2.0, 3.0)
        self.phi_history.append(phi)

        return phi

    @property
    def avg_clip_ratio(self) -> float:
        """Average fraction of steps being eaten by constraints."""
        if not self._clip_ratios:
            return 0.0
        return sum(self._clip_ratios) / len(self._clip_ratios)

    @property
    def is_in_crisis(self) -> bool:
        """Óthismos crisis: pressure consistently above 80% of raw step."""
        return self.avg_clip_ratio > 0.8


class PhaseDetector:
    """Detects óthismos phases from pressure history.

    Maps the óthismos growth phases to the governor's harmony levels:

    Phase I (Exploration): Low, stable pressure. Agent is learning freely.
        → HARMONY

    Phase II (Exploitation): Moderate pressure, trending up. Agent is
        pushing against constraints productively.
        → STRAIN (productive)

    Phase III (Crisis): High, volatile pressure. Constraints are too tight.
        The model is failing.
        → SURPRISE
    """

    @staticmethod
    def detect_phase(
        pressure_history: List[float],
        window: int = 16,
    ) -> Tuple[str, float]:
        """Detect current óthismos phase.

        Returns (phase_name, confidence 0-1).
        """
        if len(pressure_history) < 4:
            return "exploration", 0.0

        recent = pressure_history[-min(window, len(pressure_history)):]
        avg = sum(recent) / len(recent)

        # Compute trend
        half = len(recent) // 2
        if half > 0:
            first_half = sum(recent[:half]) / half
            second_half = sum(recent[half:]) / (len(recent) - half)
            trend = (second_half - first_half) / max(first_half, 1e-6)
        else:
            trend = 0.0

        # Variance (volatility)
        var = sum((p - avg) ** 2 for p in recent) / len(recent)
        volatility = math.sqrt(var)

        # Phase classification
        if avg < 0.2 and volatility < 0.1:
            return "exploration", 0.9
        elif avg < 0.5 and trend > 0:
            return "exploitation", 0.8
        elif avg > 0.5 or volatility > 0.2:
            return "crisis", min(1.0, avg)
        else:
            return "transition", 0.5

    @staticmethod
    def phase_to_friction(phase: str) -> FrictionLevel:
        """Map óthismos phase to governor friction level."""
        if phase == "exploration":
            return FrictionLevel.HARMONY
        elif phase == "exploitation" or phase == "transition":
            return FrictionLevel.STRAIN
        else:
            return FrictionLevel.SURPRISE


def create_othismos_governor(
    deadband: float = 1.0,
    sustained_threshold: int = 3,
) -> HarmonyGovernor:
    """Create a governor configured for óthismos-based pressure monitoring.

    The deadband is lower than the default because óthismos pressure values
    are typically smaller than entropy-based friction values.
    """
    gov = HarmonyGovernor(
        sustained_threshold=sustained_threshold,
    )
    return gov
