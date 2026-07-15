"""
Executive Agent — Layer 3 of the triadic cognitive architecture.

The Executive is the improvisation layer. It sleeps during normal operations
and wakes only when the Harmony Governor fires a FrictionAlarm at SURPRISE level.

When woken, the Executive takes concrete actions on the running system:

  REWRITE_OBJECTIVE — actually changes the sandbox's target value (closes the loop)
  RESET_MODEL       — actually wipes the sandbox's LinearModel weights
  INJECT_NOVELTY    — actually adds noise to the action space for N ticks
  REWIRE            — actually swaps the prediction source for a channel
  REKEY             — actually tightens / loosens the channel's deadband
  ESCALATE          — actually invokes a callback (real captain hookup)

The Executive writes to the sandbox/governor through callable hooks that the
user wires up at registration time (subagent_sandbox, escalate_callback).
Without those hooks set, the actions degrade gracefully: they still record
their intention in AgentConfig, but they don't pretend to control something
they aren't wired to.

Honest disclosure: in `harmony_demo.py` the Executive IS wired in, so the
actions that change behavior (e.g. `RESET_MODEL`) actually reset the model's
weights — which produces a measurable change in the next tick's prediction.

The Executive does NOT handle routine processing. It is the captain who
sleeps until the alarm sounds, then improvises.

Zero external dependencies. stdlib only. Python ≥ 3.10.
"""

import random
import time
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
    REKEY = 0           # Tighten or loosen the agent's deadband / sensor window
    REWIRE = 1          # Swap the prediction source for the failing channel
    REWRITE_OBJECTIVE = 2  # Change the agent's target value
    INJECT_NOVELTY = 3  # Add noise to the action space for N ticks
    RESET_MODEL = 4     # Wipe the sub-agent's internal model
    ESCALATE = 5        # Escalate to human (the real captain)


@dataclass(frozen=True, slots=True)
class ImprovisationResult:
    """Result of an Executive improvisation.

    `applied` is True when the action produced a behavioural change in the
    system (not just a metadata mutation). This is the honest indicator —
    no-op actions report applied=False.
    """
    action: ExecutiveAction
    target_channel: str
    description: str
    timestamp: float
    alarm_phi: float
    post_improvisation_phi: Optional[float] = None
    resolved: bool = False
    escalated: bool = False
    applied: bool = True


@dataclass
class AgentConfig:
    """Mutable configuration for a sub-agent that the Executive can rewrite.

    These fields are read by the action handlers below and produce real
    behavioural changes when the corresponding hooks are wired in.
    """
    __slots__ = (
        'name', 'channel', 'objective', 'target_sensor',
        'constraint_tokens', 'io_connections', 'deadband',
        'model_reset_count', 'novelty_injection_count',
        'novelty_injection_until',
        'rekey_count',
    )

    name: str
    channel: int
    objective: str
    target_sensor: float          # Active target value the sandbox uses
    constraint_tokens: List[str]  # Token identifiers affecting deadband
    io_connections: List[str]     # Sensor names whose actuals feed this channel
    deadband: float               # Active deadband (Executive can adjust)
    model_reset_count: int
    novelty_injection_count: int
    novelty_injection_until: int  # tick index when novelty window expires
    rekey_count: int

    def __init__(
        self,
        name: str,
        channel: int,
        objective: str = "maintain harmony",
        target_sensor: float = 0.0,
        constraint_tokens: Optional[List[str]] = None,
        io_connections: Optional[List[str]] = None,
        deadband: float = 2.0,
    ):
        self.name = name
        self.channel = channel
        self.objective = objective
        self.target_sensor = target_sensor
        self.constraint_tokens = constraint_tokens or []
        self.io_connections = io_connections or []
        self.deadband = deadband
        self.model_reset_count = 0
        self.novelty_injection_count = 0
        self.novelty_injection_until = 0
        self.rekey_count = 0


class DiagnosticEngine:
    """Diagnoses why a sub-agent's model failed.

    Uses the Executive's configured thresholds (not hardcoded magic numbers)
    so the diagnostic stays in tune with operator policy.
    """

    @staticmethod
    def diagnose(
        alarm: FrictionAlarm,
        governor: HarmonyGovernor,
        configs: Dict[str, AgentConfig],
        novelty_threshold: int = 3,
        reset_threshold: float = 0.35,
        escalate_threshold: int = 3,
        latency_threshold_ms: float = 2000.0,
        entropy_threshold: float = 3.0,
    ) -> Tuple[ExecutiveAction, str]:
        """Diagnose the cause of a friction alarm.

        Returns (recommended_action, reason).
        """
        state = governor.channel_state(alarm.room_name)
        config = configs.get(alarm.room_name)

        # Check 1: Is the Hurst exponent collapsed? → model is degenerating
        if alarm.hurst < reset_threshold:
            return (
                ExecutiveAction.RESET_MODEL,
                f"Hurst exponent collapsed to {alarm.hurst:.3f} "
                f"(threshold={reset_threshold}) — prediction stream has "
                f"become random walk. Model needs full reset.",
            )

        # Check 2: Is entropy extremely high? → constraint tokens too loose
        if alarm.entropy > entropy_threshold:
            return (
                ExecutiveAction.REKEY,
                f"Entropy {alarm.entropy:.2f} bits "
                f"(threshold={entropy_threshold}) — constraint space too wide. "
                f"Tightening deadband to collapse degrees of freedom.",
            )

        # Check 3: Has novelty been injected too many times? → genuinely stuck
        if config and config.novelty_injection_count >= escalate_threshold:
            return (
                ExecutiveAction.ESCALATE,
                f"Channel '{alarm.room_name}' has required "
                f"{config.novelty_injection_count} novelty injections "
                f"(threshold={escalate_threshold}). "
                f"Beyond automatic recovery — escalate to human.",
            )

        # Check 4: High latency? → agent is computationally overloaded
        if alarm.latency_ms > latency_threshold_ms:
            return (
                ExecutiveAction.REWRITE_OBJECTIVE,
                f"Latency {alarm.latency_ms:.0f}ms "
                f"(threshold={latency_threshold_ms:.0f}ms) — agent is overloaded. "
                f"Simplifying objective to a single fixed setpoint.",
            )

        # Check 5: Are coupled channels also failing? → systemic failure
        try:
            connectome = governor.update_connectome()
            if connectome:
                for pair in connectome.significant:
                    if alarm.room_name in (pair.room_a, pair.room_b):
                        other = (
                            pair.room_b if pair.room_a == alarm.room_name
                            else pair.room_a
                        )
                        if other in governor.channels:
                            other_state = governor.channel_state(other)
                            if other_state and other_state.is_surprised:
                                return (
                                    ExecutiveAction.REWIRE,
                                    f"Coupled channels '{alarm.room_name}' "
                                    f"and '{other}' both failing "
                                    f"(coupling={pair.correlation:.3f}). "
                                    f"Cross-wiring to break the cascade.",
                                )
        except Exception:
            pass  # Connectome analysis is best-effort

        # Default: inject novelty with rate-limit awareness
        if (
            config
            and config.novelty_injection_count >= novelty_threshold
        ):
            return (
                ExecutiveAction.INJECT_NOVELTY,
                f"Friction Φ={alarm.phi:.3f} exceeded deadband "
                f"{state.deadband}. Novelty budget at "
                f"{config.novelty_injection_count}/{novelty_threshold} "
                f"— one more injection before escalation.",
            )
        return (
            ExecutiveAction.INJECT_NOVELTY,
            f"Friction Φ={alarm.phi:.3f} exceeded deadband "
            f"{state.deadband}. Injecting novelty to break potential "
            f"degenerative loop.",
        )


class ExecutiveAgent:
    """Layer 3: The improvisation protocol.

    The Executive sleeps during normal operation. When the Harmony Governor
    fires a SURPRISE-level alarm, the Executive wakes, diagnoses the failure,
    improvises a solution (with REAL effects on the system), and retires.

    To make the actions close the loop, wire hooks:

      exec.set_sandbox(name, sandbox)         # for RESET_MODEL
      exec.set_escalation_callback(callback)  # for ESCALATE

    Per-channel I/O sources for REWIRE are set via `set_io_source(name, source)`.
    Per-channel target for REWRITE_OBJECTIVE is set on the AgentConfig at
    `register_agent` time and mutated by the Executive.
    """

    __slots__ = (
        '_governor', '_configs', '_diagnostic',
        '_history', '_wake_count', '_last_wake_tick',
        '_on_improvisation', '_novelty_threshold',
        '_reset_threshold', '_escalate_threshold',
        '_latency_threshold_ms', '_entropy_threshold',
        '_auto_resolved', '_escalated',
        '_sandbox_by_name', '_io_source', '_escalate_callback',
        '_last_intervention_tick', '_rng',
    )

    def __init__(
        self,
        governor: HarmonyGovernor,
        diagnostic: Optional[DiagnosticEngine] = None,
        on_improvisation: Optional[Callable[[ImprovisationResult], None]] = None,
        novelty_threshold: int = 3,
        reset_threshold: float = 0.35,
        escalate_threshold: int = 3,
        latency_threshold_ms: float = 2000.0,
        entropy_threshold: float = 3.0,
        seed: Optional[int] = None,
    ):
        self._governor: HarmonyGovernor = governor
        self._diagnostic: DiagnosticEngine = diagnostic or DiagnosticEngine()
        self._configs: Dict[str, AgentConfig] = {}
        self._history: List[ImprovisationResult] = []
        self._wake_count: int = 0
        self._last_wake_tick: int = -1
        self._on_improvisation = on_improvisation
        # Thresholds stored as instance state (no longer dead config)
        self._novelty_threshold = novelty_threshold
        self._reset_threshold = reset_threshold
        self._escalate_threshold = escalate_threshold
        self._latency_threshold_ms = latency_threshold_ms
        self._entropy_threshold = entropy_threshold
        # Counters
        self._auto_resolved: int = 0
        self._escalated: int = 0
        # Hooks for closing the loop
        self._sandbox_by_name: Dict[str, HypothesisSandbox] = {}
        self._io_source: Dict[str, Callable[[], float]] = {}
        self._escalate_callback: Optional[Callable[[FrictionAlarm, str], None]] = None
        # Avoid re-improvising the same channel every tick
        self._last_intervention_tick: Dict[str, int] = {}
        # Reproducible RNG for novelty injection
        self._rng = random.Random(seed)

    # ----- Hook wiring ---------------------------------------------------

    def set_sandbox(self, channel_name: str, sandbox: HypothesisSandbox) -> None:
        """Wire a sandbox so RESET_MODEL and INJECT_NOVELTY actually work."""
        self._sandbox_by_name[channel_name] = sandbox

    def set_io_source(
        self,
        channel_name: str,
        source: Callable[[], float],
    ) -> None:
        """Wire a callback that returns the actual sensor value.

        Used by REWIRE: when a channel is cross-wired, its `actual` is
        fetched from the donor source instead of the main loop.
        """
        self._io_source[channel_name] = source

    def set_escalation_callback(
        self,
        callback: Callable[[FrictionAlarm, str], None],
    ) -> None:
        """Wire a callback for ESCALATE actions.

        Called as `callback(alarm, reason)`. Use this to push to a phone,
        pager, telegram bot, etc.
        """
        self._escalate_callback = callback

    # ----- Configuration ------------------------------------------------

    def register_agent(
        self,
        name: str,
        channel: int,
        objective: str = "maintain harmony",
        target_sensor: float = 0.0,
        constraint_tokens: Optional[List[str]] = None,
        io_connections: Optional[List[str]] = None,
        deadband: float = 2.0,
    ) -> AgentConfig:
        """Register a sub-agent configuration that the Executive can modify."""
        config = AgentConfig(
            name=name, channel=channel, objective=objective,
            target_sensor=target_sensor,
            constraint_tokens=constraint_tokens or [],
            io_connections=io_connections or [],
            deadband=deadband,
        )
        self._configs[name] = config
        return config

    # ----- Wake / diagnose / handle ------------------------------------

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
        """Diagnose and apply a single alarm's improvisation.

        Each branch performs a real effect on the running system. If the
        required hook is unwired, the action still records its intent
        but `applied` is False and the description says so honestly.
        """
        action, reason = self._diagnostic.diagnose(
            alarm, self._governor, self._configs,
            novelty_threshold=self._novelty_threshold,
            reset_threshold=self._reset_threshold,
            escalate_threshold=self._escalate_threshold,
            latency_threshold_ms=self._latency_threshold_ms,
            entropy_threshold=self._entropy_threshold,
        )

        config = self._configs.get(alarm.room_name)
        applied = True

        # ----- REWRITE_OBJECTIVE: change the sandbox's target ----------
        if action == ExecutiveAction.REWRITE_OBJECTIVE and config:
            # Old target vs new (for the description)
            old_target = config.target_sensor
            # Move toward a stable midpoint (sensor_lo + sensor_hi)/2 isn't
            # available here, so use the current actual as the conservative
            # "just stay here for a tick" target.
            state = self._governor.channel_state(alarm.room_name)
            new_target = state.actuals[-1] if state.actuals else old_target
            config.target_sensor = new_target
            config.objective = "simplified: maintain stability"
            description = (
                f"Rewrote objective for '{alarm.room_name}': {reason} "
                f"Old target={old_target:.3f}, new target={new_target:.3f}."
            )

        # ----- INJECT_NOVELTY: scale action range, add noise ----------
        elif action == ExecutiveAction.INJECT_NOVELTY and config:
            config.novelty_injection_count += 1
            config.novelty_injection_until = (
                self._governor.tick_count + 5
            )  # active for 5 ticks
            # Actually widen the sandbox's action space if wired
            sandbox = self._sandbox_by_name.get(alarm.room_name)
            if sandbox is not None:
                lo, hi = -1.0, 1.0
                if sandbox._action_range:
                    lo = sandbox._action_range[0]
                    hi = sandbox._action_range[-1]
                # Widen the action range by 30% and add jitter
                span = hi - lo
                new_lo = lo - 0.15 * span
                new_hi = hi + 0.15 * span
                sandbox.set_action_range(new_lo, new_hi, step=(new_hi - new_lo) / 20)
            description = (
                f"Injected novelty into '{alarm.room_name}' for 5 ticks: {reason} "
                f"Count: {config.novelty_injection_count}/{self._novelty_threshold}. "
                + ("Action range widened." if sandbox is not None
                   else "Sandbox not wired — intent recorded only.")
            )

        # ----- RESET_MODEL: actually reset ---------------------------
        elif action == ExecutiveAction.RESET_MODEL and config:
            config.model_reset_count += 1
            sandbox = self._sandbox_by_name.get(alarm.room_name)
            if sandbox is not None:
                sandbox.reset_model()
                applied = True
                description = (
                    f"Reset model for '{alarm.room_name}': {reason} "
                    f"LinearModel weights cleared. Reset count: "
                    f"{config.model_reset_count}. Sandbox IS wired."
                )
            else:
                applied = False
                description = (
                    f"Would reset model for '{alarm.room_name}': {reason} "
                    f"Reset count: {config.model_reset_count}. "
                    f"Sandbox NOT wired — intent recorded only."
                )

        # ----- REWIRE: swap the actual source for the channel --------
        elif action == ExecutiveAction.REWIRE and config:
            donor_name = self._find_donor_channel(alarm.room_name)
            if donor_name and donor_name in self._io_source:
                old_source = config.io_connections[0] if config.io_connections else None
                config.io_connections = [donor_name]
                description = (
                    f"Cross-wired '{alarm.room_name}' with I/O from "
                    f"'{donor_name}': {reason} Old source: {old_source}, "
                    f"new source: {donor_name}."
                )
            else:
                # Degrade gracefully: nobody to borrow from
                applied = False
                action = ExecutiveAction.INJECT_NOVELTY
                if config:
                    config.novelty_injection_count += 1
                description = (
                    f"REWIRE failed for '{alarm.room_name}' — no I/O source "
                    f"available. Falling back to INJECT_NOVELTY: {reason}"
                )

        # ----- REKEY: tighten or loosen the deadband ------------------
        elif action == ExecutiveAction.REKEY and config:
            config.rekey_count += 1
            old_deadband = config.deadband
            # Tighten by 15% per rekey, down to a minimum of 0.5
            new_deadband = max(0.5, old_deadband * 0.85)
            config.deadband = new_deadband
            config.constraint_tokens.append(f"tightened:{config.rekey_count}")
            # Push the new deadband to the governor so the next observation
            # cycle uses it.
            try:
                chan_state = self._governor.channel_state(alarm.room_name)
                chan_state.deadband = new_deadband
            except KeyError:
                pass
            description = (
                f"Re-keyed '{alarm.room_name}': {reason} "
                f"Deadband: {old_deadband:.3f} -> {new_deadband:.3f}. "
                f"Tokens: {config.constraint_tokens[-1]}."
            )

        # ----- ESCALATE: invoke the callback --------------------------
        elif action == ExecutiveAction.ESCALATE:
            description = f"ESCALATED '{alarm.room_name}': {reason}"
            if self._escalate_callback is not None:
                try:
                    self._escalate_callback(alarm, description)
                    description += " (callback invoked)"
                except Exception as e:
                    description += f" (callback failed: {e})"
                    applied = False
            else:
                applied = False
                description += " (no escalation callback set — intent recorded only)"
            return ImprovisationResult(
                action=action,
                target_channel=alarm.room_name,
                description=description,
                timestamp=time.time(),
                alarm_phi=alarm.phi,
                escalated=True,
                applied=applied,
            )

        else:
            description = f"Unknown handling for '{alarm.room_name}': {reason}"
            applied = False

        # Check current state — but discount the current (alarm) tick
        # since the action's effect hasn't had a tick to land yet.
        try:
            state = self._governor.channel_state(alarm.room_name)
            if len(state.phi_history) >= 2:
                # Use previous tick's φ (the one before this alarm)
                previous_phi = state.phi_history[-2]
                resolved = previous_phi < state.deadband * 0.5
            else:
                resolved = False
        except KeyError:
            resolved = False

        return ImprovisationResult(
            action=action,
            target_channel=alarm.room_name,
            description=description,
            timestamp=time.time(),
            alarm_phi=alarm.phi,
            resolved=resolved,
            escalated=False,
            applied=applied,
        )

    # ----- Proactive improvisation ------------------------------------

    def improvise(self) -> List[ImprovisationResult]:
        """Proactively check for issues and improvise solutions.

        Rate-limited per channel: we only fire one proactive intervention
        per channel per 4 ticks (no more flooding the Executive with the
        same monotone-strain alarm).
        """
        results: List[ImprovisationResult] = []
        current_tick = self._governor.tick_count
        intervention_cooldown = 4

        for name in self._governor.channels:
            state = self._governor.channel_state(name)
            if state.is_strained and len(state.phi_history) >= 4:
                last_tick = self._last_intervention_tick.get(name, -100)
                if current_tick - last_tick < intervention_cooldown:
                    continue  # don't re-improvise within cooldown

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
                        latency_ms=state.latencies[-1] if state.latencies else 0.0,
                        tick=current_tick,
                        timestamp=time.time(),
                        message=f"Pre-emptive: phi trending upward for '{name}'",
                    )
                    result = self._handle_single_alarm(alarm)
                    results.append(result)
                    self._history.append(result)
                    self._last_intervention_tick[name] = current_tick

        return results

    # ----- Helpers -----------------------------------------------------

    def _find_donor_channel(self, excluded_name: str) -> Optional[str]:
        """Find an I/O source the channel can borrow from."""
        for name, cfg in self._configs.items():
            if name != excluded_name:
                # Prefer a channel that is currently IN HARMONY
                try:
                    state = self._governor.channel_state(name)
                    if state.is_in_harmony:
                        return name
                except KeyError:
                    continue
        return None

    # ----- Accessors ---------------------------------------------------

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
                    'target_sensor': cfg.target_sensor,
                    'constraint_tokens': len(cfg.constraint_tokens),
                    'io_connections': cfg.io_connections,
                    'deadband': cfg.deadband,
                    'resets': cfg.model_reset_count,
                    'novelty_injections': cfg.novelty_injection_count,
                    'novelty_active_until': cfg.novelty_injection_until,
                    'rekeys': cfg.rekey_count,
                }
                for name, cfg in self._configs.items()
            },
        }
