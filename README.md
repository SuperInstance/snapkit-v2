# SnapKit v2 — Eisenstein Lattice Snap, Temporal, Spectral, Connectome

Constraint geometry snap toolkit — Eisenstein Voronoï, temporal beat grids, spectral analysis, connectome detection, and FLUX-Tensor-MIDI timing.

**Zero external dependencies. stdlib only. Python ≥ 3.10.**

## Why Eisenstein?

The Eisenstein integers ℤ[ω] (where ω = e^(2πi/3)) form the A₂ root lattice — the densest possible packing in 2D. This gives:

- **12-fold symmetry** (6 rotations × 2 reflections)
- **Optimal covering** — minimizes maximum distance from any point to its nearest lattice point
- **PID property** — H¹ = 0 guarantee for sheaf-theoretic consistency
- **Hexagonal Voronoï cells** — isotropic quantization error

When you snap continuous values to this lattice, you get the tightest possible discrete approximation in 2D. No other lattice does better.

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
print(f"Eisenstein integer: {e}, norm²={e.norm_squared}")

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

# Snap timestamps to the beat grid
result = snap.observe(t=1.04, value=0.3)
print(f"On beat: {result.is_on_beat}, offset: {result.offset:.3f}")
# On beat: True, offset: 0.040

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

# Individual functions
from snapkit import entropy, hurst_exponent, autocorrelation
h = entropy(signal, bins=10)
H = hurst_exponent(signal)
acf = autocorrelation(signal, max_lag=50)
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

# Export to Graphviz
print(result.to_graphviz())

# Adjacency matrix
names, matrix = result.adjacency_matrix()
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

# Tempo changes
flux.tempo.set_tempo(tick=960, bpm=140)
seconds = flux.tempo.tick_to_seconds(1920)
```

## API Reference

### `snapkit.eisenstein` — Eisenstein Lattice

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `EisensteinInteger` | `(a: int, b: int)` | Frozen dataclass: a + bω on the A₂ lattice |
| `EisensteinInteger.complex` | property | Convert to Cartesian complex |
| `EisensteinInteger.norm_squared` | property | a² - ab + b² (always ≥ 0) |
| `EisensteinInteger.from_complex(z)` | classmethod | Round complex → nearest Eisenstein integer |
| `eisenstein_round(z)` | `(complex) → EisensteinInteger` | True nearest via Voronoï cell |
| `eisenstein_round_naive(z)` | `(complex) → EisensteinInteger` | Legacy 4-candidate rounding |
| `eisenstein_snap(z, tol=0.5)` | `(complex, float) → (EI, float, bool)` | Snap with tolerance check |
| `eisenstein_snap_batch(pts, tol)` | `(list, float) → list` | Vectorized snap |
| `eisenstein_distance(z1, z2)` | `(complex, complex) → float` | Lattice distance |
| `eisenstein_fundamental_domain(z)` | `(complex) → (unit, EI)` | Reduce to canonical representative |

Arithmetic operators: `+`, `-`, `*`, `conjugate()`, `abs()`.

### `snapkit.eisenstein_voronoi` — Voronoï Cell Snap

| Symbol | Description |
|--------|-------------|
| `eisenstein_snap_voronoi(x, y)` | True nearest-neighbor via A₂ Voronoï cell (optimized: squared distance) |
| `eisenstein_snap_naive(x, y)` | Fast approximate snap |
| `eisenstein_snap_batch(points)` | Vectorized Voronoï snap |
| `eisenstein_to_real(a, b)` | Convert (a, b) → (x, y) Cartesian |
| `snap_distance(x, y, a, b)` | Euclidean distance to lattice point |

### `snapkit.temporal` — Beat Grid & T-minus-0

| Symbol | Description |
|--------|-------------|
| `BeatGrid(period, phase, t_start)` | Periodic time grid |
| `BeatGrid.snap(t, tolerance)` | Snap timestamp → `TemporalResult` |
| `BeatGrid.snap_batch(timestamps, tol)` | Vectorized snap |
| `BeatGrid.nearest_beat(t)` | `(beat_time, beat_index)` |
| `BeatGrid.beats_in_range(t_start, t_end)` | All beats in interval |
| `TemporalSnap(grid, tolerance, t0_threshold, t0_window)` | Beat snap + T-minus-0 zero-crossing detection |
| `TemporalSnap.observe(t, value)` | Feed observation → `TemporalResult` |
| `TemporalResult` | Frozen: `original_time`, `snapped_time`, `offset`, `is_on_beat`, `is_t_minus_0`, `beat_index`, `beat_phase` |

### `snapkit.spectral` — Signal Analysis

| Symbol | Description |
|--------|-------------|
| `entropy(data, bins=10)` | Shannon entropy via histogram binning |
| `hurst_exponent(data)` | R/S analysis Hurst exponent (H ≈ 0.5 = random, H > 0.5 = trending, H < 0.5 = mean-reverting) |
| `autocorrelation(data, max_lag)` | Normalized autocorrelation function |
| `spectral_summary(data, bins, max_lag)` | Full summary → `SpectralSummary` |
| `spectral_batch(series_list, bins, max_lag)` | Batch spectral analysis |
| `SpectralSummary` | `entropy_bits`, `hurst`, `autocorr_lag1`, `autocorr_decay`, `is_stationary` |

### `snapkit.connectome` — Room Coupling

| Symbol | Description |
|--------|-------------|
| `TemporalConnectome(threshold, max_lag, min_samples)` | Coupled/anti-coupled room detection via cross-correlation |
| `TemporalConnectome.add_room(name, activity)` | Register a room's activity trace |
| `TemporalConnectome.analyze()` | → `ConnectomeResult` |
| `ConnectomeResult.coupled` | List of positively coupled pairs |
| `ConnectomeResult.anti_coupled` | List of negatively coupled pairs |
| `ConnectomeResult.significant` | All non-uncoupled pairs |
| `ConnectomeResult.adjacency_matrix()` | `(names, matrix)` |
| `ConnectomeResult.to_graphviz()` | Graphviz DOT string |
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
| `TempoMap(ticks_per_beat, initial_bpm)` | Tick ↔ seconds conversion with tempo changes |
| `Room(name, channel, voice)` | Musician/channel with note helpers |
| `MIDIEvent` | Frozen: `tick`, `channel`, `event_type`, `value`, `velocity` |

## Performance

- Voronoï snap uses **squared-distance comparison** (no `sqrt` in hot path)
- `BeatGrid` uses **precomputed inverse period** (`1/period`)
- Autocorrelation uses **local variable caching** and precomputed `inv_r0`
- All dataclasses use `__slots__` / `frozen=True`
- Batch operations available on all modules

## Connection to Constraint Theory

SnapKit v2 is the production core of the Cocapn constraint theory system:

- **Eisenstein lattice** provides the optimal 2D quantization surface (A₂ root system)
- **Temporal snap** connects to the FLUX-Tensor timing protocol for multi-room musical coordination
- **Spectral analysis** detects self-similarity (Hurst exponent) and entropy for snap calibration
- **Connectome** detects coupled and anti-coupled rooms for the constraint network
- **MIDI module** implements the FLUX-Tensor-MIDI protocol for temporal constraint enforcement

## License

MIT

---

*Part of the Cocapn constraint theory ecosystem.*
