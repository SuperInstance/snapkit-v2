#!/usr/bin/env python3
"""
snapkit CLI — Harmony governor dashboard and utilities.

Usage:
    snapkit harmony          Run interactive harmony monitor
    snapkit demo             Run the integration demo
    snapkit lattice          Show the maritime token lattice
    snapkit info             Show package info
"""

import argparse
import sys
import os
import time
import math

# Ensure package is importable when run from source
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_info(args):
    """Show package info."""
    import snapkit
    print(f"snapkit v{snapkit.__version__}")
    print(f"Modules: {', '.join(sorted(m for m in snapkit.__all__ if m[0].isupper()))}")
    print()
    print("Architecture layers:")
    print("  Layer 1: HypothesisSandbox (forward simulation + óthismos scoring)")
    print("  Layer 2: HarmonyGovernor (FEP friction monitoring + alarms)")
    print("  Layer 3: ExecutiveAgent (improvisation protocol)")
    print("  I/O:     MIDIBridge (sensor → MIDI bus wiring)")
    print("  Tokens:  TokenLattice (Eisenstein constraint space)")
    print()
    print("pip install cocapn-snapkit")
    print("https://github.com/SuperInstance/snapkit-v2")


def cmd_lattice(args):
    """Show the maritime token lattice."""
    from snapkit.clever_tokens import create_maritime_lattice, ConstraintType

    lattice = create_maritime_lattice()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Maritime Clever Token Lattice (Eisenstein A₂)       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    type_symbols = {
        ConstraintType.RIGID: "■",
        ConstraintType.ELASTIC: "◆",
        ConstraintType.PERIODIC: "◉",
        ConstraintType.COUPLED: "⬡",
        ConstraintType.DIRECTIVE: "→",
    }

    for tid in lattice.tokens:
        token = lattice.get_token(tid)
        sym = type_symbols.get(token.constraint_type, "?")
        dev = ""
        if token.directive_target:
            dev = f" → target ({token.directive_target.a},{token.directive_target.b})"
        coup = f" ⇄ {token.coupled_with}" if token.coupled_with else ""

        print(f"  {sym} {tid}")
        print(f"    lattice: ({token.lattice_point.a}, {token.lattice_point.b})"
              f"  snap: {token.snap_radius:.2f}"
              f"  E={token.expected_entropy:.1f}b H={token.expected_hurst:.2f}"
              f"{dev}{coup}")
        if token.metadata:
            print(f"    note: {token.metadata}")
        print()

    # Show pairwise distances
    distances = lattice.lattice_distances()
    if distances:
        print("Lattice distances (constraint density):")
        for (a, b), d in sorted(distances.items(), key=lambda x: x[1]):
            print(f"  {a} ↔ {b}: {d:.2f}")
        print()

    # Render a sample prompt
    print("Sample constraint block (helm:steady + nav:harbor_approach):")
    print("─" * 50)
    print(lattice.render_prompt(["helm:steady", "nav:harbor_approach"]))
    print("─" * 50)


def cmd_harmony(args):
    """Interactive harmony monitor."""
    from snapkit.governor import HarmonyGovernor, FrictionLevel
    from snapkit.midi_io import MIDIBridge

    gov = HarmonyGovernor(
        beat_period=args.period,
        sustained_threshold=args.threshold,
    )
    gov.register_channel("helm", channel=0, deadband=args.deadband)

    bridge = MIDIBridge(governor=gov)
    bridge.register_sensor("heading", lo=0, hi=360, source="nmea2000")
    bridge.register_sensor("rpm", lo=0, hi=3000, source="nmea2000")

    print("╔══════════════════════════════════════════════════════════╗")
    print("║          HARMONY MONITOR — Live Friction Dashboard      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("  Press Ctrl+C to stop")
    print(f"  Deadband: {args.deadband}  Period: {args.period}s  Threshold: {args.threshold}")
    print()

    tick = 0
    try:
        while True:
            gov.tick(tick)

            # Simulated sensor feed (in production: read from serial/MQTT)
            import random
            heading = 180.0 + 10 * math.sin(tick * 0.1) + random.gauss(0, 1)
            rpm = 1800 + random.gauss(0, 20)

            bridge.feed_sensor("heading", heading)
            bridge.feed_sensor("rpm", rpm)

            # Simulated prediction (in production: from HypothesisSandbox)
            predicted_heading = 180.0 + 8 * math.sin(tick * 0.1)
            gov.record_observation(
                "helm",
                prediction=predicted_heading,
                actual=heading,
                latency_ms=random.uniform(5, 15),
            )

            # Dashboard
            state = gov.channel_state("helm")
            phi = state.phi
            harmony = "✓ HARMONY" if phi < args.deadband * 0.7 else \
                      "△ STRAIN" if phi < args.deadband else \
                      "✗ SURPRISE"

            tempo = bridge.tempo_deriver

            print(
                f"\r  t={tick:4d} | HDG={heading:6.1f}° "
                f"RPM={rpm:4.0f} "
                f"Φ={phi:.3f} {harmony} "
                f"BPM={tempo.bpm:.1f} "
                f"Alarms={len(gov.unacknowledged_alarms)}",
                end="", flush=True,
            )

            if gov.unacknowledged_alarms:
                print()
                for alarm in gov.alarms:
                    print(f"\n  🚨 {alarm.room_name}: Φ={alarm.phi:.3f} "
                          f"E={alarm.entropy:.2f} H={alarm.hurst:.2f}")
                gov.clear_alarms()

            tick += 1
            time.sleep(args.period)

    except KeyboardInterrupt:
        print("\n\n  Stopped.")
        print(f"  Ticks: {tick}")
        print(f"  Global Φ: {gov.global_phi():.3f}")
        print(f"  MIDI events: {bridge._flux.event_count}")
        print(f"  Tempo: {tempo.bpm:.1f} BPM (stability: {tempo.stability:.2f})")


def cmd_demo(args):
    """Run the integration demo."""
    demo_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "examples", "harmony_demo.py",
    )
    if not os.path.exists(demo_path):
        print("Demo not found. Run from the snapkit-v2 repo:")
        print("  python3 examples/harmony_demo.py")
        sys.exit(1)

    import runpy
    runpy.run_path(demo_path, run_name="__main__")


def main():
    parser = argparse.ArgumentParser(
        prog="snapkit",
        description="snapkit v2 — Harmony governance for multi-agent systems",
    )
    sub = parser.add_subparsers(dest="command")

    # info
    sub.add_parser("info", help="Show package info")

    # lattice
    sub.add_parser("lattice", help="Show the maritime token lattice")

    # demo
    sub.add_parser("demo", help="Run the integration demo")

    # harmony
    p_harm = sub.add_parser("harmony", help="Interactive harmony monitor")
    p_harm.add_argument("--period", type=float, default=1.0,
                        help="Beat period in seconds (default: 1.0)")
    p_harm.add_argument("--deadband", type=float, default=2.0,
                        help="Friction deadband (default: 2.0)")
    p_harm.add_argument("--threshold", type=int, default=3,
                        help="Sustained surprise count before alarm (default: 3)")

    args = parser.parse_args()

    if args.command == "info":
        cmd_info(args)
    elif args.command == "lattice":
        cmd_lattice(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "harmony":
        cmd_harmony(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
