"""
Executive Agent — Layer 3 of the triadic cognitive architecture.

The Executive is the improvisation layer. It sleeps during normal operations
and wakes only when the Harmony Governor fires a FrictionAlarm at SURPRISE level.

When woken, the Executive can:
  1. Diagnose the failed state space (why did the sub-agent's model break?)
  2. Rewrite constraint tokens (change the key / re-tune the agent)
  3. Cross-wire previously unrelated I/O streams
  4. Alter objective functions
  5. Inject novelty to break degenerative loops
  6. Retire (go quiet again) once harmony is restored

The Executive does NOT handle routine processing. It is the captain who
sleeps until the alarm sounds, then improvises.

Zero external dependencies. stdlib only. Python ≥ 3.10.
"""

import time
import math
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from snapkit.governor import (
    HarmonyGovernor, FrictionAlarm, FrictionLevel, ChannelState,
)
from snapkit.sandbox import HypothesisSandbox, SandboxScore


class ExecutiveAction(IntEnum):
    """Types of improvisation the Executive can perform."""
    REKEY = 0           # Change the agent's constraint tokens (re-tune)
    REWIRE = 1          # Cross-wire I/O streams between agents
    REWRITE_OBJECTIVE = 2  # Change the agent's goal function
    INJECT_NOVELTY = 3  # Inject noise/novelty to break degenerative loop
    RESET_MODEL = 4     # Reset the sub-agent's internal model
    ESCALATE = 5        # Escalate to human (the real captain)


@dataclass(frozen=True, slots=True)
class ImprovisationResult:
    """Result of an Executive improvisation."""
    action: ExecutiveAction
    target_channel: str
    description: str
    timestamp: float
    alarm_phi: float
    post_improvisation_phi: Optional[float] = None
    resolved: bool = False
    escalated: bool = False


@dataclass
class AgentConfig:
    """Mutable configuration for a sub-agent that the Executive can rewrite."""
    __slots__ = (
        'name', 'channel', 'objective', 'constraint_tokens',
        'io_connections', 'deadband', 'model_reset_count',
        'novelty_injection_count', 'rekey_count',
    )

    name: str
    channel: int
    objective: str
    constraint_tokens: List[str]
    io_connections: List[str]
    deadband: float
    model_reset_count: int
    novelty_injection_count: int
    rekey_count: int

    def __init__(
        self,
        name: str,
        channel: int,
        objective: str = "maintain harmony",
        constraint_tokens: Optional[List[str]] = None,
        io_connections: Optional[List[str]] = None,
        deadband: float = 2.0,
    ):
        self.name = name
        self.channel = channel
        self.objective = objective
        self.constraint_tokens = constraint_tokens or []
        self.io_connections = io_connections or []
        self.deadband = deadband
        self.model_reset_count = 0
        self.novelty_injection_count = 0
        self.rekey_count = 0


class DiagnosticEngine:
    """Diagnoses why a sub-agent's model failed.

    Examines the friction alarm and the governor state to determine
    the most likely cause of failure and the best improvisation strategy.
    """

    @staticmethod
    def diagnose(
        alarm: FrictionAlarm,
        governor: HarmonyGovernor,
        configs: Dict[str, AgentConfig],
    ) -> Tuple[ExecutiveAction, str]:
        """Diagnose the cause of a friction alarm.

        Returns (recommended_action, reason).
        """
        state = governor.channel_state(alarm.room_name)
        config = configs.get(alarm.room_name)

        # Check 1: Is the Hurst exponent collapsed? → model is degenerating
        if alarm.hurst < 0.35:
            return (
                ExecutiveAction.RESET_MODEL,
                f"Hurst exponent collapsed to {alarm.hurst:.3f} — "
                f"prediction stream has become random walk. "
                f"Model needs full reset.",
            )

        # Check 2: Is entropy extremely high? → constraint tokens too loose
        if alarm.entropy > 3.0:
            return (
                ExecutiveAction.REKEY,
                f"Entropy {alarm.entropy:.2f} bits — constraint space too wide. "
                f"Need tighter clever tokens to collapse degrees of freedom.",
            )

        # Check 3: Has novelty been injected recently? → might be degenerative loop
        if config and config.novelty_injection_count > 3:
            return (
                ExecutiveAction.ESCALATE,
                f"Channel '{alarm.room_name}' has required "
                f"{config.novelty_injection_count} novelty injections. "
                f"This is beyond automatic recovery — escalate to human.",
            )

        # Check 4: High latency? → agent is computationally overloaded
        if alarm.latency_ms > 2000:
            return (
                ExecutiveAction.REWRITE_OBJECTIVE,
                f"Latency {alarm.latency_ms:.0f}ms — agent is overloaded. "
                f"Simplify objective function to reduce computational burden.",
            )

        # Check 5: Are coupled channels also failing? → systemic failure
        connectome = governor.update_connectome()
        if connectome:
            for pair in connectome.significant:
                if alarm.room_name in (pair.room_a, pair.room_b):
                    other = pair.room_b if pair.room_a == alarm.room_name else pair.room_a
                    other_state = governor.channel_state(other) if other in governor.channels else None
                    if other_state and other_state.is_surprised:
                        return (
                            ExecutiveAction.REWIRE,
                            f"Coupled channels '{alarm.room_name}' and '{other}' "
                            f"both failing (coupling={pair.correlation:.3f}). "
                            f"Systemic failure — cross-wire to alternate I/O.",
                        )

        # Default: inject novelty
        return (
            ExecutiveAction.INJECT_NOVELTY,
            f"Friction Φ={alarm.phi:.3f} exceeded deadband {state.deadband}. "
            f"Injecting novelty to break potential degenerative loop.",
        )


class ExecutiveAgent:
    """Layer 3: The improvisation protocol.

    The Executive sleeps during normal operation. When the Harmony Governor
    fires a SURPRISE-level alarm, the Executive wakes, diagnoses the failure,
    improvises a solution, and retires.

    Usage:
        governor = HarmonyGovernor()
        governor.register_channel("helm", channel=0)
        executive = ExecutiveAgent(governor)
        executive.register_agent("helm", channel=0, objective="maintain course")

        # When governor fires alarms:
        results = executive.handle_alarms()

        # Or proactively:
        if executive.should_wake():
            results = executive.improvise()
    """

    __slots__ = (
        '_governor', '_configs', '_diagnostic',
        '_history', '_wake_count', '_last_wake_tick',
        '_on_improvisation', '_novelty_threshold',
        '_reset_threshold', '_escalate_threshold',
        '_auto_resolved', '_escalated',
    )

    def __init__(
        self,
        governor: HarmonyGovernor,
        diagnostic: Optional[DiagnosticEngine] = None,
        on_improvisation: Optional[Callable[[ImprovisationResult], None]] = None,
        novelty_threshold: int = 3,
        reset_threshold: int = 2,
        escalate_threshold: int = 3,
    ):
        self._governor: HarmonyGovernor = governor
        self._diagnostic: DiagnosticEngine = diagnostic or DiagnosticEngine()
        self._configs: Dict[str, AgentConfig] = {}
        self._history: List[ImprovisationResult] = []
        self._wake_count: int = 0
        self._last_wake_tick: int = -1
        self._on_improvisation = on_improvisation
        self._novelty_threshold = novelty_threshold
        self._reset_threshold = reset_threshold
        self._escalate_threshold = escalate_threshold
        self._auto_resolved: int = 0
        self._escalated: int = 0

    def register_agent(
        self,
        name: str,
        channel: int,
        objective: str = "maintain harmony",
        constraint_tokens: Optional[List[str]] = None,
        io_connections: Optional[List[str]] = None,
        deadband: float = 2.0,
    ) -> AgentConfig:
        """Register a sub-agent configuration that the Executive can modify."""
        config = AgentConfig(
            name=name, channel=channel, objective=objective,
            constraint_tokens=constraint_tokens or [],
            io_connections=io_connections or [],
            deadband=deadband,
        )
        self._configs[name] = config
        return config

    def should_wake(self) -> bool:
        """Check if the Executive should wake up."""
        return len(self._governor.unacknowledged_alarms) > 0

    def handle_alarms(self) -> List[ImprovisationResult]:
        """Process all unacknowledged alarms.

        The Executive wakes, diagnoses each alarm, performs the
        improvisation, and checks if harmony was restored.
        """
        if not self.should_wake():
            return []

        self._wake_count += 1
        self._last_wake_tick = self._governor.tick_count

        alarms = self._governor.unacknowledged_alarms
        results: List[ImprovisationResult] = []

        for alarm in alarms:
            result = self._handle_single_alarm(alarm)
            results.append(result)
            self._history.append(result)

            if result.resolved and not result.escalated:
                self._auto_resolved += 1
            elif result.escalated:
                self._escalated += 1

            if self._on_improvisation:
                self._on_improvisation(result)

        self._governor.clear_alarms()
        return results

    def _handle_single_alarm(self, alarm: FrictionAlarm) -> ImprovisationResult:
        """Diagnose and resolve a single alarm."""
        action, reason = self._diagnostic.diagnose(
            alarm, self._governor, self._configs,
        )

        config = self._configs.get(alarm.room_name)

        if action == ExecutiveAction.REKEY and config:
            config.rekey_count += 1
            # Tighten constraints — add a more specific token
            new_token = f"constraint:v2:{config.rekey_count}"
            config.constraint_tokens.append(new_token)
            description = f"Re-keyed '{alarm.room_name}': {reason} Added {new_token}."

        elif action == ExecutiveAction.REWIRE and config:
            # Cross-wire to another agent's I/O
            other_channels = [
                n for n in self._configs
                if n != alarm.room_name and self._configs[n].io_connections
            ]
            if other_channels:
                donor = other_channels[0]
                borrowed = self._configs[donor].io_connections[:1]
                config.io_connections.extend(borrowed)
                description = (
                    f"Cross-wired '{alarm.room_name}' with I/O from '{donor}': "
                    f"{reason} Borrowed: {borrowed}."
                )
            else:
                action = ExecutiveAction.INJECT_NOVELTY
                config.novelty_injection_count += 1
                description = f"No donor available for rewiring. {reason}"

        elif action == ExecutiveAction.REWRITE_OBJECTIVE and config:
            config.objective = f"simplified: maintain stability"
            description = (
                f"Rewrote objective for '{alarm.room_name}': {reason} "
                f"New objective: 'maintain stability'."
            )

        elif action == ExecutiveAction.INJECT_NOVELTY and config:
            config.novelty_injection_count += 1
            description = (
                f"Injected novelty into '{alarm.room_name}': {reason} "
                f"Count: {config.novelty_injection_count}."
            )

        elif action == ExecutiveAction.RESET_MODEL and config:
            config.model_reset_count += 1
            description = (
                f"Reset model for '{alarm.room_name}': {reason} "
                f"Reset count: {config.model_reset_count}."
            )

        elif action == ExecutiveAction.ESCALATE:
            description = f"ESCALATED '{alarm.room_name}': {reason}"
            return ImprovisationResult(
                action=action,
                target_channel=alarm.room_name,
                description=description,
                timestamp=time.time(),
                alarm_phi=alarm.phi,
                escalated=True,
            )

        else:
            description = f"Unknown handling for '{alarm.room_name}': {reason}"

        # Check if the channel has recovered after improvisation
        # (In a real system, we'd wait for the next tick. Here we check current state.)
        state = self._governor.channel_state(alarm.room_name)
        resolved = state.phi < state.deadband * 0.5  # Back in solid harmony

        return ImprovisationResult(
            action=action,
            target_channel=alarm.room_name,
            description=description,
            timestamp=time.time(),
            alarm_phi=alarm.phi,
            resolved=resolved,
            escalated=False,
        )

    def improvise(self) -> List[ImprovisationResult]:
        """Proactively check for issues and improvise solutions.

        Even without explicit alarms, the Executive can detect
        patterns that warrant intervention.
        """
        results: List[ImprovisationResult] = []

        # Check for channels trending toward surprise
        for name, state in self._governor._channels.items():
            if state.is_strained and len(state.phi_history) >= 4:
                # Check if phi is trending upward
                recent = list(state.phi_history)[-4:]
                if all(recent[i] <= recent[i + 1] for i in range(len(recent) - 1)):
                    # Phi is monotonically increasing — pre-emptive intervention
                    alarm = FrictionAlarm(
                        channel=state.channel,
                        room_name=name,
                        level=FrictionLevel.STRAIN,
                        phi=state.phi,
                        entropy=state.last_summary.entropy_bits if state.last_summary else 0,
                        hurst=state.last_summary.hurst if state.last_summary else 0.5,
                        latency_ms=sum(state.latencies) / max(len(state.latencies), 1),
                        tick=self._governor.tick_count,
                        timestamp=time.time(),
                        message=f"Pre-emptive: phi trending upward for '{name}'",
                    )
                    result = self._handle_single_alarm(alarm)
                    results.append(result)
                    self._history.append(result)

        return results

    @property
    def wake_count(self) -> int:
        return self._wake_count

    @property
    def history(self) -> List[ImprovisationResult]:
        return list(self._history)

    @property
    def auto_resolved(self) -> int:
        return self._auto_resolved

    @property
    def escalated(self) -> int:
        return self._escalated

    def config(self, name: str) -> AgentConfig:
        return self._configs[name]

    def state(self) -> Dict:
        """Snapshot of the Executive state."""
        return {
            'wake_count': self._wake_count,
            'last_wake_tick': self._last_wake_tick,
            'registered_agents': list(self._configs.keys()),
            'total_improvisations': len(self._history),
            'auto_resolved': self._auto_resolved,
            'escalated': self._escalated,
            'pending_alarms': len(self._governor.unacknowledged_alarms),
            'agents': {
                name: {
                    'objective': cfg.objective,
                    'constraint_tokens': len(cfg.constraint_tokens),
                    'io_connections': cfg.io_connections,
                    'deadband': cfg.deadband,
                    'resets': cfg.model_reset_count,
                    'novelty_injections': cfg.novelty_injection_count,
                    'rekeys': cfg.rekey_count,
                }
                for name, cfg in self._configs.items()
            },
        }
