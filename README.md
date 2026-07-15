# snapkit-v2

The Architecture of Harmony — a triadic cognitive architecture for multi-agent systems based on the Free Energy Principle.

## What is this?

snapkit-v2 is a constraint-geometry toolkit that implements FEP (Free Energy Principle) cognition for AI agents. Each agent operates on the Eisenstein A₂ lattice, communicates via MIDI-style temporal events, and is monitored by a Harmony Governor that measures "friction" — the degree to which an agent's internal model fails to predict its sensory inputs.

## The Triadic Architecture

```
┌─────────────────────────────────────────┐
│  Layer 3: Executive (Agency)            │
│  Wakes on friction alarm, improvises    │
│  (rewrites constraints, cross-wires I/O)│
└──────────────────┬──────────────────────┘
                   │ tuning forks
                   ▼
┌─────────────────────────────────────────┐
│  Layer 2: Harmony Governor              │
│  Measures cognitive friction (Φ)        │
│  Triggers Executive when Φ > deadband   │
└──────┬───────────────────────┬──────────┘
       │ MIDI ch 0             │ MIDI ch N
       ▼                       ▼
┌──────────────┐      ┌──────────────┐
│  Layer 1     │      │  Layer 1     │
│  Sandbox     │      │  Sandbox     │
│  Forward     │      │  Forward     │
│  simulation  │      │  simulation  │
└──────────────┘      └──────────────┘
```

## Install

```bash
pip install cocapn-snapkit
```

From source:
```bash
git clone https://github.com/SuperInstance/snapkit-v2
cd snapkit-v2
pip install -e .
```

For MIDI hardware support:
```bash
pip install cocapn-snapkit[midi]
```

## Quick Start

### Run the integration demo

```bash
python3 examples/harmony_demo.py
```

Simulates a fishing boat in calm → rough seas. Shows the full triadic architecture in action.

### View the maritime token lattice

```bash
snapkit lattice
```

### Run the harmony monitor

```bash
snapkit harmony --period 1.0 --deadband 1.5
```

### Listen to the harmony

```python
from snapkit.audio import harmony_demo_audio
harmony_demo_audio("/tmp/harmony.wav")
```

### Use as a library

```python
from snapkit.governor import HarmonyGovernor
from snapkit.sandbox import HypothesisSandbox
from snapkit.executive import ExecutiveAgent
from snapkit.midi_io import MIDIBridge
from snapkit.clever_tokens import create_maritime_lattice

# Set up the triadic architecture
gov = HarmonyGovernor()
gov.register_channel("helm", channel=0)

sandbox = HypothesisSandbox(sensor_name="heading")
sandbox.set_action_range(-1.0, 1.0, step=0.1)

executive = ExecutiveAgent(gov)
executive.register_agent("helm", channel=0)

bridge = MIDIBridge(governor=gov)
bridge.register_sensor("heading", lo=0, hi=360)

# In your main loop:
gov.tick()
phi = sandbox.evaluate(sensor_current=heading, target_sensor=target)
sandbox.observe(action_taken=best_action, sensor_before=h, sensor_after=h2)
gov.record_observation("helm", prediction=p, actual=h2)

# When something breaks:
results = executive.handle_alarms()
```

## Physical Hardware

### ESP32 + MPU6050 IMU

See `firmware/esp32_mpu6050_imu/`. Flash the Arduino sketch, connect to a serial port, and feed IMU data to the MIDI bridge:

```python
import serial, json
from snapkit.midi_io import MIDIBridge

bridge = MIDIBridge(governor=gov)
ser = serial.Serial('/dev/ttyUSB0', 115200)

while True:
    line = ser.readline()
    data = json.loads(line)
    if 'roll' in data:
        bpm = bridge.feed_roll(data['roll'])
        print(f"Hull tempo: {bpm:.1f} BPM")
```

## Architecture Layers

| Module | Purpose |
|--------|---------|
| `eisenstein.py` | Geometric constraint space (A₂ lattice) |
| `temporal.py` | BeatGrid, T-minus-0 detection |
| `spectral.py` | Entropy, Hurst exponent, autocorrelation |
| `connectome.py` | Coupled/anti-coupled room detection |
| `midi.py` | FluxTensorMIDI protocol |
| `midi_io.py` | Bridge to physical sensors |
| `clever_tokens.py` | Lattice-anchored constraint tokens |
| `sandbox.py` | Layer 1: Forward simulation + óthismos scoring |
| `governor.py` | Layer 2: FEP friction monitoring |
| `executive.py` | Layer 3: Improvisation protocol |
| `fleet.py` | Multi-vessel coordination |
| `othismos_bridge.py` | Connects to the othismos library |
| `audio.py` | Synthesize MIDI bus as audio (listen to harmony) |
| `cli.py` | Command-line tools |

## Web Dashboard

Open `examples/harmony_dashboard.html` in a browser for a real-time view of:
- Per-channel friction state
- Hull-derived tempo
- MIDI piano roll
- Event log

For a live deployment, serve the dashboard from a Fleet Coordinator:
```bash
docker run -p 8000:8000 snapkit-v2
```

## JavaScript / TypeScript

The core primitives are available in TypeScript at `js/snapkit.ts` for browser-side use:

```typescript
import { HarmonyGovernor, BeatGrid, TokenLattice } from './snapkit.js';

const gov = new HarmonyGovernor();
gov.registerChannel('helm', 0, 1.5);
gov.tick();
gov.recordObservation('helm', 180.0, 182.0);
console.log(gov.systemState());
```

## Related Repos

- [othismos](https://github.com/SuperInstance/othismos) — Constraint pressure theory
- [constraint-theory-core](https://github.com/SuperInstance/constraint-theory-core) — Algebraic primitives
- [style-dna](https://github.com/SuperInstance/style-dna) — Musical DNA extraction
- [snapkit-js](https://github.com/SuperInstance/snapkit-js) — Full JS port
- [spline-midi-smooth](https://github.com/SuperInstance/spline-midi-smooth) — MIDI interpolation
- [AI-Writings](https://github.com/SuperInstance/AI-Writings) — Reflections and essays

## Documentation

- [Architecture of Harmony](docs/ARCHITECTURE_OF_HARMONY.md) — The full white paper
- [ESP32 firmware](firmware/esp32_mpu6050_imu/README.md) — Physical sensor layer
- [Integration demo](examples/harmony_demo.py) — Worked example

## License

MIT

---

*The hull sets the beat. The agents sync to the ocean.*