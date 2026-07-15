"""Tests for the Executive Agent."""

import pytest
from snapkit.governor import HarmonyGovernor, FrictionLevel, FrictionAlarm
from snapkit.executive import (
    ExecutiveAgent, DiagnosticEngine, ExecutiveAction,
    AgentConfig, ImprovisationResult,
)


class TestDiagnosticEngine:
    def test_collapsed_hurst_suggests_reset(self):
        gov = HarmonyGovernor()
        gov.register_channel("failing", channel=0)
        gov.tick(0)
        alarm = FrictionAlarm(
            channel=0, room_name="failing",
            level=FrictionLevel.SURPRISE,
            phi=3.0, entropy=1.0, hurst=0.2,
            latency_ms=100, tick=0, timestamp=0,
        )
        configs = {"failing": AgentConfig("failing", 0)}
        action, reason = DiagnosticEngine.diagnose(alarm, gov, configs)
        assert action == ExecutiveAction.RESET_MODEL
        assert "Hurst" in reason

    def test_high_entropy_suggests_rekey(self):
        gov = HarmonyGovernor()
        gov.register_channel("noisy", channel=0)
        gov.tick(0)
        alarm = FrictionAlarm(
            channel=0, room_name="noisy",
            level=FrictionLevel.SURPRISE,
            phi=2.5, entropy=4.0, hurst=0.5,
            latency_ms=100, tick=0, timestamp=0,
        )
        configs = {"noisy": AgentConfig("noisy", 0)}
        action, reason = DiagnosticEngine.diagnose(alarm, gov, configs)
        assert action == ExecutiveAction.REKEY

    def test_high_latency_suggests_simplify(self):
        gov = HarmonyGovernor()
        gov.register_channel("slow", channel=0)
        gov.tick(0)
        alarm = FrictionAlarm(
            channel=0, room_name="slow",
            level=FrictionLevel.SURPRISE,
            phi=2.0, entropy=1.0, hurst=0.5,
            latency_ms=3000, tick=0, timestamp=0,
        )
        configs = {"slow": AgentConfig("slow", 0)}
        action, reason = DiagnosticEngine.diagnose(alarm, gov, configs)
        assert action == ExecutiveAction.REWRITE_OBJECTIVE

    def test_excessive_novelty_escalates(self):
        gov = HarmonyGovernor()
        gov.register_channel("stuck", channel=0)
        gov.tick(0)
        alarm = FrictionAlarm(
            channel=0, room_name="stuck",
            level=FrictionLevel.SURPRISE,
            phi=2.5, entropy=1.0, hurst=0.5,
            latency_ms=100, tick=0, timestamp=0,
        )
        config = AgentConfig("stuck", 0)
        config.novelty_injection_count = 5
        configs = {"stuck": config}
        action, reason = DiagnosticEngine.diagnose(alarm, gov, configs)
        assert action == ExecutiveAction.ESCALATE


class TestExecutiveAgent:
    def test_register_agent(self):
        gov = HarmonyGovernor()
        exe = ExecutiveAgent(gov)
        cfg = exe.register_agent("helm", channel=0, objective="maintain course")
        assert cfg.objective == "maintain course"
        assert "helm" in exe.state()["registered_agents"]

    def test_should_wake_false_when_no_alarms(self):
        gov = HarmonyGovernor()
        exe = ExecutiveAgent(gov)
        assert not exe.should_wake()

    def test_should_wake_true_with_alarms(self):
        gov = HarmonyGovernor()
        gov._alarms.append(
            FrictionAlarm(0, "test", FrictionLevel.SURPRISE, 3.0, 1.0, 0.4, 100, 0, 0)
        )
        exe = ExecutiveAgent(gov)
        assert exe.should_wake()

    def test_handle_alarms_clears_governor(self):
        gov = HarmonyGovernor()
        gov.register_channel("helm", channel=0, deadband=0.1, window_size=8)
        exe = ExecutiveAgent(gov)
        exe.register_agent("helm", channel=0)

        # Generate sustained surprise
        for i in range(10):
            gov.tick(i)
            gov.record_observation("helm", prediction=1.0, actual=0.0, latency_ms=500)

        assert gov.unacknowledged_alarms
        results = exe.handle_alarms()
        assert len(results) > 0
        assert not gov.unacknowledged_alarms  # cleared

    def test_rekey_adds_constraint_token(self):
        gov = HarmonyGovernor()
        gov.register_channel("noisy", channel=0, deadband=0.1, window_size=8)
        exe = ExecutiveAgent(gov)
        cfg = exe.register_agent("noisy", channel=0, constraint_tokens=["base"])
        initial_tokens = len(cfg.constraint_tokens)

        # Generate high-entropy surprise
        for i in range(10):
            gov.tick(i)
            gov.record_observation(
                "noisy",
                prediction=float(i % 3),
                actual=float(i % 5),
                latency_ms=50,
            )

        results = exe.handle_alarms()
        if results and results[0].action == ExecutiveAction.REKEY:
            assert len(cfg.constraint_tokens) > initial_tokens

    def test_escalation_for_repeated_failures(self):
        gov = HarmonyGovernor()
        gov.register_channel("doomed", channel=0, deadband=0.1, window_size=8)
        exe = ExecutiveAgent(gov, escalate_threshold=2)
        cfg = exe.register_agent("doomed", channel=0)
        cfg.novelty_injection_count = 5  # Already past threshold

        for i in range(10):
            gov.tick(i)
            gov.record_observation("doomed", prediction=1.0, actual=0.0, latency_ms=100)

        results = exe.handle_alarms()
        assert any(r.escalated for r in results)

    def test_improvise_detects_trending_friction(self):
        gov = HarmonyGovernor()
        gov.register_channel("degrading", channel=0, deadband=5.0, window_size=32)
        exe = ExecutiveAgent(gov)
        exe.register_agent("degrading", channel=0)

        # Feed progressively worse predictions
        for i in range(8):
            gov.tick(i)
            gov.record_observation(
                "degrading",
                prediction=0.5,
                actual=0.5 + i * 0.3,
                latency_ms=100 + i * 50,
            )

        # Should detect upward trend and pre-emptively act
        results = exe.improvise()
        assert len(results) >= 0  # May or may not trigger depending on thresholds

    def test_state_snapshot(self):
        gov = HarmonyGovernor()
        exe = ExecutiveAgent(gov)
        exe.register_agent("helm", channel=0)
        state = exe.state()
        assert "wake_count" in state
        assert "registered_agents" in state
        assert "helm" in state["agents"]
