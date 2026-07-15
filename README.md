# SnapKit v2 — Eisenstein Lattice Snap, Temporal, Spectral, Connectome, FLUX-Tensor-MIDI

Constraint geometry snap toolkit for Python. Snaps continuous 2D points to the **Eisenstein A₂ lattice** (densest 2D packing), provides temporal beat-grid alignment, spectral analysis, connectome (room coupling) detection, FLUX-Tensor-MIDI timing, **Harmony Governor** (FEP friction monitoring), and **Hypothesis Sandbox** (forward simulation + óthismos scoring). **Zero external dependencies. stdlib only. Python ≥ 3.10.**

## Why Eisenstein?

The Eisenstein integers ℤ[ω] (ω = e^(2πi/3)) form the A₂ root lattice — hexagonal grid, densest possible packing in 2D:

- **12-fold symmetry** (6 rotations × 2 reflections)
- **Optimal covering** — minimizes max distance from any point to its nearest lattice point
- **PID property** — H¹ = 0 guarantee for sheaf-theoretic consistency
- **Isotropic error** — hexagonal Voronoï cells spread quantization evenly

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

## Quick Start

### Eisenstein Lattice Snap

```python
from snapkit import EisensteinInteger, eisenstein_snap, eisenstein_round

# Snap a complex number to the nearest Eisenstein integer
z = complex(0.3, 0.7)
nearest, distance, is_snap = eisenstein_snap(z, tolerance=0.5)
print(f"{nearest} — distance={distance:.4f}, snapped={is_snap}")
# EisensteinInteger(0, 1) — distance=0.1339, snapped=True

# Round directly
e = EisensteinInteger.from_complex(z)

# Arithmetic
a = EisensteinInteger(3, 1)
b = EisensteinInteger(1, 2)
print(a + b)          # EisensteinInteger(4, 3)
print(a * b)          # EisensteinInteger(1, 7)
print(a.conjugate())  # EisensteinInteger(4, -1)
```

### Temporal Snap (Beat Grid + T-minus-0)

```python
from snapkit import BeatGrid, TemporalSnap

grid = BeatGrid(period=1.0, phase=0.0)
snap = TemporalSnap(grid, tolerance=0.1, t0_threshold=0.05)

result = snap.observe(t=1.04, value=0.3)
print(f"On beat: {result.is_on_beat}, offset: {result.offset:.3f}")

# T-minus-0 detection: zero-crossing in value derivatives
result = snap.observe(t=2.01, value=0.001)
print(f"T-0 detected: {result.is_t_minus_0}")
```

### Spectral Analysis

```python
from snapkit import spectral_summary
import random

signal = [random.gauss(0, 1) for _ in range(500)]
summary = spectral_summary(signal)
print(f"Entropy: {summary.entropy_bits:.2f} bits")
print(f"Hurst: {summary.hurst:.3f} (stationary: {summary.is_stationary})")
print(f"ACF lag-1: {summary.autocorr_lag1:.3f}, decay: {summary.autocorr_decay}")
```

### Connectome (Room Coupling Detection)

```python
from snapkit import TemporalConnectome

conn = TemporalConnectome(threshold=0.3, max_lag=5)
conn.add_room("alpha", [0.1, 0.5, 0.3, 0.8, 0.2])
conn.add_room("beta",  [0.2, 0.4, 0.4, 0.7, 0.3])
conn.add_room("gamma", [0.9, 0.1, 0.7, 0.2, 0.8])

result = conn.analyze()
for pair in result.significant:
    print(f"{pair.room_a} ↔ {pair.room_b}: {pair.coupling.value} (r={pair.correlation:.3f}, lag={pair.lag})")

print(result.to_graphviz())               # Graphviz DOT output
names, matrix = result.adjacency_matrix() # Correlation matrix
```

### FLUX-Tensor-MIDI

```python
from snapkit import FluxTensorMIDI, TempoMap

flux = FluxTensorMIDI(TempoMap(ticks_per_beat=480, initial_bpm=120))
piano = flux.add_room("piano", channel=0)
drums = flux.add_room("drums", channel=9)

flux.note_on("piano", tick=0, note=60, velocity=100)
flux.note_off("piano", tick=480, note=60)
flux.note_on("drums", tick=0, note=36)

events = flux.render()       # sorted by tick
quantized = flux.quantize(grid=120)  # snap to 16th note grid
flux.tempo.set_tempo(tick=960, bpm=140)
seconds = flux.tempo.tick_to_seconds(1920)
```

## API Reference

### `snapkit.eisenstein` — Lattice Operations

| Symbol | Description |
|--------|-------------|
| `EisensteinInteger(a, b)` | Frozen dataclass on the A₂ lattice |
| `EisensteinInteger.complex` | Cartesian complex representation |
| `EisensteinInteger.norm_squared` | a² − ab + b² (always ≥ 0) |
| `EisensteinInteger.from_complex(z)` | Round → nearest Eisenstein integer |
| `eisenstein_round(z)` | True nearest via Voronoï cell |
| `eisenstein_round_naive(z)` | Legacy 4-candidate rounding |
| `eisenstein_snap(z, tol=0.5)` | Snap with tolerance check → `(EI, float, bool)` |
| `eisenstein_snap_batch(pts, tol)` | Vectorized snap |
| `eisenstein_distance(z1, z2)` | Lattice distance |
| `eisenstein_fundamental_domain(z)` | Reduce to canonical representative |

Arithmetic: `+`, `-`, `*`, `conjugate()`, `abs()`.

### `snapkit.eisenstein_voronoi` — Voronoï Cell Snap

| Symbol | Description |
|--------|-------------|
| `eisenstein_snap_voronoi(x, y)` | True nearest-neighbor (squared distance, no sqrt) |
| `eisenstein_snap_naive(x, y)` | Fast approximate snap |
| `eisenstein_snap_batch(points)` | Vectorized Voronoï |
| `eisenstein_to_real(a, b)` | (a, b) → (x, y) Cartesian |
| `snap_distance(x, y, a, b)` | Euclidean distance to lattice point |

### `snapkit.temporal` — Beat Grid & T-minus-0

| Symbol | Description |
|--------|-------------|
| `BeatGrid(period, phase, t_start)` | Periodic time grid |
| `BeatGrid.snap(t, tolerance)` | Snap → `TemporalResult` |
| `BeatGrid.snap_batch(timestamps, tol)` | Vectorized snap |
| `BeatGrid.nearest_beat(t)` | `(beat_time, beat_index)` |
| `BeatGrid.beats_in_range(t_start, t_end)` | All beats in interval |
| `TemporalSnap(grid, tolerance, t0_threshold, t0_window)` | Beat snap + zero-crossing detection |
| `TemporalSnap.observe(t, value)` | Feed observation → `TemporalResult` |

### `snapkit.spectral` — Signal Analysis

| Symbol | Description |
|--------|-------------|
| `entropy(data, bins=10)` | Shannon entropy via histogram |
| `hurst_exponent(data)` | R/S analysis (H ≈ 0.5 = random, > 0.5 = trending, < 0.5 = mean-reverting) |
| `autocorrelation(data, max_lag)` | Normalized autocorrelation |
| `spectral_summary(data, bins, max_lag)` | → `SpectralSummary` |
| `spectral_batch(series_list, bins, max_lag)` | Batch analysis |

### `snapkit.connectome` — Room Coupling

| Symbol | Description |
|--------|-------------|
| `TemporalConnectome(threshold, max_lag, min_samples)` | Cross-correlation coupling detection |
| `TemporalConnectome.add_room(name, activity)` | Register room activity trace |
| `TemporalConnectome.analyze()` | → `ConnectomeResult` |
| `ConnectomeResult.coupled` / `.anti_coupled` / `.significant` | Coupled pair lists |
| `ConnectomeResult.adjacency_matrix()` | `(names, matrix)` |
| `ConnectomeResult.to_graphviz()` | DOT string |
| `RoomPair` | `room_a`, `room_b`, `coupling`, `correlation`, `lag`, `confidence` |
| `CouplingType` | `COUPLED`, `ANTI_COUPLED`, `UNCOUPLED` |

### `snapkit.midi` — FLUX-Tensor-MIDI

| Symbol | Description |
|--------|-------------|
| `FluxTensorMIDI(tempo_map)` | Conductor: rooms, events, quantize, render |
| `FluxTensorMIDI.add_room(name, channel, voice)` | Register a room (musician) |
| `FluxTensorMIDI.note_on(room, tick, note, velocity)` | Schedule note-on |
| `FluxTensorMIDI.note_off(room, tick, note)` | Schedule note-off |
| `FluxTensorMIDI.render()` | All events sorted by tick |
| `FluxTensorMIDI.quantize(grid)` | Snap events to grid |
| `TempoMap(ticks_per_beat, initial_bpm)` | Tick ↔ seconds with tempo changes |

## Performance

- Voronoï snap uses **squared-distance comparison** (no `sqrt` in hot path)
- `BeatGrid` uses **precomputed inverse period** (`1/period`)
- Autocorrelation uses **local variable caching** and precomputed `inv_r0`
- All dataclasses use `__slots__` / `frozen=True`
- Batch operations available on all modules

## Connection to Constraint Theory

SnapKit v2 is the production core of the Cocapn constraint theory system:

- **Eisenstein lattice** — optimal 2D quantization (A₂ root system)
- **Temporal snap** — FLUX-Tensor timing for multi-room coordination
- **Spectral analysis** — self-similarity (Hurst) and entropy for snap calibration
- **Connectome** — coupled/anti-coupled room detection for the constraint network
- **MIDI** — FLUX-Tensor-MIDI protocol for temporal constraint enforcement

## Documentation

- [User Guide](docs/USER-GUIDE.md) — Complete usage documentation

## Related Repos

- [snapkit-js](https://github.com/SuperInstance/snapkit-js) — JavaScript/TypeScript version (Eisenstein + temporal + spectral)
- [constraint-theory-core](https://github.com/SuperInstance/constraint-theory-core) — Mathematical primitives
- [style-dna](https://github.com/SuperInstance/style-dna) — Musical DNA extraction and style morphing
- [spline-midi-smooth](https://github.com/SuperInstance/spline-midi-smooth) — Spline interpolation for MIDI automation
- [copilot-for-eclipse](https://github.com/SuperInstance/copilot-for-eclipse) — Constraint Theory MCP for Copilot in Eclipse

## License

MIT
