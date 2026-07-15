"""Tests for óthismos bridge."""

import pytest
from snapkit.othismos_bridge import (
    PressureGaugeChannel, PhaseDetector,
    create_othismos_governor,
)
from snapkit.governor import HarmonyGovernor, FrictionLevel


class TestPressureGaugeChannel:
    def test_record_pressure(self):
        gauge = PressureGaugeChannel("training", channel=0)
        phi = gauge.record_pressure(step=0, pressure=0.1, raw_step_norm=1.0)
        assert phi >= 0.0
        assert len(gauge._pressure_history) == 1

    def test_high_pressure_high_phi(self):
        gauge = PressureGaugeChannel("training", channel=0)
        for i in range(10):
            gauge.record_pressure(
                step=i, pressure=0.8, raw_step_norm=1.0,
            )
        assert gauge.phi > 0.5

    def test_low_pressure_low_phi(self):
        gauge = PressureGaugeChannel("training", channel=0)
        for i in range(10):
            gauge.record_pressure(
                step=i, pressure=0.01, raw_step_norm=1.0,
            )
        assert gauge.phi < 0.5

    def test_clip_ratio(self):
        gauge = PressureGaugeChannel("training", channel=0)
        gauge.record_pressure(step=0, pressure=0.5, raw_step_norm=1.0)
        assert gauge.avg_clip_ratio == 0.5

    def test_crisis_detection(self):
        gauge = PressureGaugeChannel("training", channel=0)
        for i in range(10):
            gauge.record_pressure(
                step=i, pressure=0.9, raw_step_norm=1.0,
            )
        assert gauge.is_in_crisis


class TestPhaseDetector:
    def test_exploration_phase(self):
        # Low, stable pressure
        pressures = [0.05, 0.06, 0.04, 0.05, 0.05, 0.06, 0.04, 0.05]
        phase, conf = PhaseDetector.detect_phase(pressures)
        assert phase == "exploration"
        assert conf > 0.5

    def test_crisis_phase(self):
        # High, volatile pressure
        pressures = [0.8, 0.3, 0.9, 0.5, 0.85, 0.4, 0.7, 0.9]
        phase, conf = PhaseDetector.detect_phase(pressures)
        assert phase == "crisis"

    def test_phase_to_friction_mapping(self):
        assert PhaseDetector.phase_to_friction("exploration") == FrictionLevel.HARMONY
        assert PhaseDetector.phase_to_friction("exploitation") == FrictionLevel.STRAIN
        assert PhaseDetector.phase_to_friction("crisis") == FrictionLevel.SURPRISE

    def test_empty_history(self):
        phase, conf = PhaseDetector.detect_phase([])
        assert phase == "exploration"
        assert conf == 0.0


class TestCreateOthismosGovernor:
    def test_creates_governor(self):
        gov = create_othismos_governor(deadband=1.0)
        assert isinstance(gov, HarmonyGovernor)
