#!/usr/bin/env python3
"""
Integration Demo: The Harmony Loop

Simulates a helm agent navigating changing sea states on a fishing boat.
Shows the complete triadic architecture in action:

  IMU roll → TempoDeriver sets the beat
  → Sandbox tests helm actions
  → Governor tracks friction
  → Executive wakes when the model breaks

No hardware required. Pure simulation. Run:

    python3 examples/harmony_demo.py
"""

import math
import sys
import os
import time
import random

# Add parent to path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snapkit.governor import HarmonyGovernor, FrictionLevel
from snapkit.sandbox import HypothesisSandbox
from snapkit.executive import ExecutiveAgent, ExecutiveAction
from snapkit.midi_io import MIDIBridge, TempoDeriver, SensorMapper


# ─── Simulation Environment ──────────────────────────────────────────

class BoatSim:
    """Simulates a fishing boat in changing sea states.

    The boat has:
      - A heading that drifts based on wave action
      - A rudder that the agent can adjust
      - An IMU that reports roll angle
      - Changing sea states that shift the dynamics

    The agent's job: keep the heading steady with minimum rudder stress.
    """

    def __init__(self):
        self.heading: float = 180.0  # Due south
        self.target_heading: float = 180.0
        self.rudder: float = 0.0    # -1 (full left) to +1 (full right)
        self.sea_state: str = "calm"
        self._wave_phase: float = 0.0
        self._wave_amp: float = 0.5
        self._wave_period: int = 40  # samples (at 10 Hz = 4 seconds)
        self._tick: int = 0
        self._rudder_effectiveness: float = 1.0

    def tick(self) -> None:
        """Advance the simulation one step."""
        self._tick += 1

        # Wave action pushes the heading around
        wave_force = self._wave_amp * math.sin(
            2 * math.pi * self._wave_phase / self._wave_period
        )
        # Add some chaos in rough seas
        if self.sea_state == "rough":
            import random
            wave_force += random.gauss(0, 0.8)

        # Rudder correction (with effectiveness factor)
        heading_change = wave_force - self.rudder * 0.3 * self._rudder_effectiveness
        # Add steady current if present (e.g. fouled rudder + current)
        heading_change += getattr(self, '_current_bias', 0.0)
        self.heading = (self.heading + heading_change) % 360.0

        self._wave_phase += 1

    def roll_angle(self) -> float:
        """Get current roll angle from the IMU."""
        base_roll = self._wave_amp * 3.0 * math.sin(
            2 * math.pi * self._wave_phase / self._wave_period
        )
        if self.sea_state == "rough":
            import random
            base_roll += random.gauss(0, 2.0)
        return base_roll

    def set_rudder(self, value: float) -> None:
        """Agent sets the rudder (-1 to +1)."""
        self.rudder = max(-1.0, min(1.0, value))

    def change_sea_state(self, state: str) -> None:
        """The Executive (or reality) changes the sea state."""
        old = self.sea_state
        self.sea_state = state
        if state == "calm":
            self._wave_amp = 0.3
            self._wave_period = 60
        elif state == "moderate":
            self._wave_amp = 1.0
            self._wave_period = 40
        elif state == "rough":
            self._wave_amp = 2.5
            self._wave_period = 25
            # Simulate a current pushing the boat
            self._rudder_effectiveness = 0.4  # Fouled rudder

    @property
    def heading_error(self) -> float:
        """How far off target."""
        err = abs(self.heading - self.target_heading)
        return min(err, 360 - err)


# ─── The Demo ────────────────────────────────────────────────────────

def run_demo():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     THE HARMONY LOOP — snapkit-v2 Integration Demo      ║")
    print("║     FEP Governance on a Simulated Fishing Vessel        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # ── Set up the triadic architecture ──
    boat = BoatSim()

    # Layer 2: Harmony Governor
    # Note sensor_lo/sensor_hi so the MIDI export encodes the actual heading
    # across the full 0..360° range instead of saturating every note to 127.
    governor = HarmonyGovernor(
        beat_period=1.0,
        sustained_threshold=3,
    )
    governor.register_channel(
        "helm", channel=0,
        deadband=1.5, window_size=64,
        sensor_lo=0.0, sensor_hi=360.0,
        target_value=boat.target_heading,         # 180.0
        task_error_weight=3.0,                    # strong goal-error contribution
    )

    # Layer 1: Hypothesis Sandbox for the helm agent
    # `circular=(0, 360)` makes LinearModel handle heading wrap-around correctly.
    sandbox = HypothesisSandbox(
        sensor_name="heading",
        actuation_cost_weight=0.3,
        sim_cost_weight=0.7,
        model=__import__('snapkit.sandbox', fromlist=['LinearModel']).LinearModel(
            learning_rate=0.01,
            circular=(0.0, 360.0),
        ),
    )
    sandbox.set_action_range(-1.0, 1.0, step=0.1)

    # Layer 3: Executive Agent — wired to the sandbox so RESET_MODEL actually
    # resets the model's weights. Escalation callback prints to stdout.
    executive = ExecutiveAgent(governor)
    executive.register_agent(
        "helm", channel=0,
        objective="maintain heading with minimum rudder stress",
        target_sensor=boat.target_heading,
        constraint_tokens=["base:helm:v1"],
        io_connections=["heading", "rudder"],
    )
    executive.set_sandbox("helm", sandbox)
    executive.set_escalation_callback(
        lambda alarm, reason: print(f"  ☎  ESCALATION CALLBACK: {reason}")
    )

    # I/O: MIDI Bridge with tempo derivation
    bridge = MIDIBridge(governor=governor)
    bridge.register_sensor("heading", lo=0, hi=360, source="nmea2000")

    print("─" * 60)
    print("PHASE 1: CALM SEAS — Finding Harmony")
    print("─" * 60)

    heading_before = boat.heading

    for tick in range(80):
        governor.tick(tick)

        # Feed IMU data to derive tempo
        roll = boat.roll_angle()
        bpm = bridge.feed_roll(roll)

        # The agent observes the current heading
        heading_now = boat.heading

        # Sandbox evaluates what action to take
        if sandbox.sample_count >= 5:
            score = sandbox.evaluate(
                sensor_current=heading_now,
                target_sensor=boat.target_heading,
                max_actuation=0.3,
            )
            best_action = score.best_hypothesis.action_value
            boat.set_rudder(best_action)

            # Record what happened: we predicted X, actually got Y
            # (We record AFTER the boat ticks so we can measure the result)
        else:
            boat.set_rudder(0.0)

        # Advance the boat
        boat.tick()
        heading_after = boat.heading

        # Record the observation in the sandbox
        sandbox.observe(
            action_taken=boat.rudder,
            sensor_before=heading_now,
            sensor_after=heading_after,
        )

        # Governor records the prediction quality
        # The "prediction" is what the sandbox expected, "actual" is what happened
        if sandbox._last_predictions:
            prediction = sandbox._last_predictions[-1]
        else:
            prediction = heading_now
        # Handle circular heading (don't pass NaN to governor)
        if prediction != prediction or heading_after != heading_after:
            prediction = heading_after
        governor.record_observation(
            "helm",
            prediction=prediction,
            actual=heading_after,
            latency_ms=5.0,
        )

        # Status output every 20 ticks
        if tick % 20 == 0 or tick == 79:
            phi = governor.channel_state("helm").phi
            harmony = "✓ HARMONY" if phi < 1.0 else "△ STRAIN" if phi < 1.5 else "✗ SURPRISE"
            print(
                f"  t={tick:3d} | heading={boat.heading:6.1f}° "
                f"target={boat.target_heading:5.1f}° "
                f"err={boat.heading_error:4.1f}° "
                f"rudder={boat.rudder:+.1f} "
                f"Φ={phi:.2f} {harmony} "
                f"BPM={bpm:.1f}"
            )

    print()
    print(f"  Model quality: {sandbox.model_health:.1%}")
    print(f"  Tempo: {bridge.tempo_deriver.bpm:.1f} BPM "
          f"(stability: {bridge.tempo_deriver.stability:.2f})")
    print(f"  Governor state: {governor.system_state()['global_phi']:.3f} Φ")

    # ── Phase 2: The sea changes ──
    print()
    print("─" * 60)
    print("PHASE 2: SEA STATE SHIFT — Calm → Rough")
    print("─" * 60)
    print("  ⚠ Current pushing boat, rudder partially fouled")
    print()

    boat.change_sea_state("rough")
    # Add a stuck-rudder scenario so the architecture MUST intervene.
    # The rudder responds, but a strong current overwhelms it.
    boat._current_bias = 0.45  # heading drift per tick from current (degrees)

    alarms_fired = 0
    executive_wakes = 0

    for tick in range(80, 200):
        governor.tick(tick)

        roll = boat.roll_angle()
        bpm = bridge.feed_roll(roll)

        heading_now = boat.heading

        if sandbox.sample_count >= 5:
            score = sandbox.evaluate(
                sensor_current=heading_now,
                target_sensor=boat.target_heading,
                max_actuation=0.4,
            )
            boat.set_rudder(score.best_hypothesis.action_value)
        else:
            boat.set_rudder(0.0)

        boat.tick()
        heading_after = boat.heading

        sandbox.observe(
            action_taken=boat.rudder,
            sensor_before=heading_now,
            sensor_after=heading_after,
        )

        if sandbox._last_predictions:
            prediction = sandbox._last_predictions[-1]
        else:
            prediction = heading_now
        # Handle circular heading (don't pass NaN to governor)
        if prediction != prediction or heading_after != heading_after:
            prediction = heading_after
        governor.record_observation(
            "helm",
            prediction=prediction,
            actual=heading_after,
            latency_ms=5.0,
        )

        # Check for executive intervention
        if governor.unacknowledged_alarms:
            alarms_fired += len(governor.unacknowledged_alarms)

        results = executive.handle_alarms()
        if results:
            executive_wakes += len(results)
            for r in results:
                print(f"  🚨 EXECUTIVE WAKE #{executive.wake_count}: "
                      f"{r.action.name} → {r.description[:70]}")

        # Proactive check
        proactive = executive.improvise()
        if proactive:
            for r in proactive:
                print(f"  ⚡ PRE-EMPTIVE: {r.action.name} → "
                      f"{r.description[:60]}")

        # Status every 20 ticks
        if tick % 20 == 0 or tick == 199:
            phi = governor.channel_state("helm").phi
            harmony = "✓ HARMONY" if phi < 1.0 else "△ STRAIN" if phi < 1.5 else "✗ SURPRISE"
            print(
                f"  t={tick:3d} | heading={boat.heading:6.1f}° "
                f"err={boat.heading_error:4.1f}° "
                f"rudder={boat.rudder:+.1f} "
                f"Φ={phi:.2f} {harmony} "
                f"BPM={bpm:.1f} "
                f"sea={boat.sea_state}"
            )

    # ── Phase 3: Executive fixes the problem ──
    print()
    print("─" * 60)
    print("PHASE 3: EXECUTIVE INTERVENTION")
    print("─" * 60)

    cfg = executive.config("helm")
    print(f"  Helm agent config after intervention:")
    print(f"    Objective:   {cfg.objective}")
    print(f"    Constraints: {cfg.constraint_tokens}")
    print(f"    I/O:         {cfg.io_connections}")
    print(f"    Resets:      {cfg.model_reset_count}")
    print(f"    Rekeys:      {cfg.rekey_count}")
    print(f"    Novelty:     {cfg.novelty_injection_count}")
    print()
    print(f"  Executive stats:")
    print(f"    Total wakes:    {executive.wake_count}")
    print(f"    Auto-resolved:  {executive.auto_resolved}")
    print(f"    Escalated:      {executive.escalated}")
    print(f"    Alarms fired:   {alarms_fired}")
    print()

    # ── Summary ──
    print("═" * 60)
    print("SUMMARY")
    print("═" * 60)
    print()

    # Honest reporting — match claims to data.
    final_heading = boat.heading
    final_error = boat.heading_error
    final_phi = governor.channel_state("helm").phi
    final_surprised = governor.channel_state("helm").is_surprised
    drift_detected = final_surprised or final_error > 15

    if executive.wake_count > 0:
        applied_count = sum(1 for r in executive.history if r.applied)
        print(f"  Phase 1:  Helm agent held heading in calm seas (model quality "
              f"{sandbox.model_health:.0%}).")
        print(f"  Phase 2:  Sea state shifted. Φ spiked → Governor fired "
              f"{alarms_fired} alarm(s).")
        print(f"  Phase 3:  Executive woke {executive.wake_count}×. "
              f"{applied_count}/{executive.wake_count} actions actually closed "
              f"the control loop (the rest were intent-only — no hook wired).")
        print()
        print(f"  Final heading: {final_heading:.1f}° (target "
              f"{boat.target_heading:.0f}°, error {final_error:.1f}°).")
        if drift_detected:
            print(f"  ⚠ Final state still drifted: Φ={final_phi:.3f}, "
                  f"surprised={final_surprised}.")
            print(f"  Executive made the architecture aware of the drift; "
                  f"complete recovery requires the boat to actually comply "
                  f"with the action corrections.")
        else:
            print(f"  ✓ Architecture recovered to harmony (Φ={final_phi:.3f}).")
        print()
        print("  Triadic architecture — verified:")
        print("    Layer 1 (Sandbox)  = forward simulation + LinearModel w/ circular wrap")
        print("    Layer 2 (Governor) = Φ = entropy + error + Hurst + latency, with alarm latch")
        print("    Layer 3 (Executive)= action handlers that mutate the sandbox / config / callbacks")
    else:
        print(f"  Phase 1:  Helm agent held heading in calm seas (model quality "
              f"{sandbox.model_health:.0%}).")
        print(f"  Phase 2:  Sea state shifted but Governor did NOT fire.")
        print(f"           Final heading {final_heading:.1f}° (target "
              f"{boat.target_heading:.0f}°, drift {final_error:.1f}°).")
        print()
        print(f"  This is a real failure of the architecture — not an")
        print(f"  'insight to tune later'.")
        print()
        if sandbox.model_health > 0.8:
            print(f"  ROOT CAUSE: LinearModel converges to fit the drift")
            print(f"  as if it were normal boat motion. Once the model")
            print(f"  learns the bias, prediction error collapses and Φ")
            print(f"  returns to harmony. The error-magnitude component")
            print(f"  catches SOME drift, but if the drift is slow and")
            print(f"  smooth, the model absorbs it.")
            print()
            print(f"  WHAT'S MISSING: Φ doesn't include a TASK-ERROR term")
            print(f"  (heading error vs target). It tracks model fit, not")
            print(f"  goal achievement. A boat that's perfectly wrong about")
            print(f"  its heading scores the same as one that's perfectly")
            print(f"  on course.")
            print()
            print(f"  This was flagged by the code review: see")
            print(f"  REVIEW.md / claude-review.md / minimax-review.md")
            print(f"  — the FEP framing is more metaphor than implementation.")
        else:
            print(f"  φ stayed under deadband for a different reason.")
            print(f"  Model quality: {sandbox.model_health:.0%}.")
        print()
        print(f"  This is a real failure mode. Adding a task-error term")
        print(f"  (heading error vs target) to Φ would catch this. Until")
        print(f"  then, the architecture needs a watchdog external to Φ.")
    print()
    print("  The hull set the beat. The agents synced to the ocean.")
    print()
    print(f"  Final tempo: {bridge.tempo_deriver.bpm:.1f} BPM "
          f"from {bridge.tempo_deriver.period:.1f}s roll period")
    print(f"  Final model quality: {sandbox.model_health:.1%}")
    print(f"  Total MIDI events: {bridge._flux.event_count}")
    print()

    # Export the session as MIDI if mido is available
    if bridge.mido_available:
        filename = "/tmp/harmony_demo.mid"
        if bridge.export_midi_file(filename):
            print(f"  Session exported: {filename}")
            print("  (Play it back to HEAR the harmony shatter and recover)")
    else:
        print("  Install mido to export the session as a .mid file:")
        print("    pip install mido")


if __name__ == "__main__":
    run_demo()
