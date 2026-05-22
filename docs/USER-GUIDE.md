# SnapKit v2 — User Guide

Complete guide to the Eisenstein lattice snap toolkit.

## Table of Contents

1. [Eisenstein Lattice](#eisenstein-lattice)
2. [EisensteinInteger](#eisensteininteger)
3. [Voronoï Cell Snap](#voronoi-cell-snap)
4. [Temporal Snap](#temporal-snap)
5. [Spectral Analysis](#spectral-analysis)
6. [Connectome](#connectome)
7. [FLUX-Tensor-MIDI](#flux-tensor-midi)
8. [Common Patterns](#common-patterns)

---

## Eisenstein Lattice

The Eisenstein integers are numbers of the form `a + bω` where `a, b ∈ ℤ` and `ω = e^(2πi/3) = -1/2 + i√3/2`.

In Cartesian coordinates:
```
x = a - b/2
y = b·√3/2
```

The lattice points form a **hexagonal grid** — the densest possible packing in 2D.

### Why This Lattice?

1. **Densest 2D packing** — Thue's theorem proves no 2D lattice packs denser
2. **12-fold symmetry** — 6 rotations × 2 reflections
3. **PID property** — ℤ[ω] is a principal ideal domain → H¹ = 0 guarantee
4. **Optimal covering** — Minimizes maximum distance from any point to its nearest lattice point
5. **Isotropic error** — Hexagonal Voronoï cells spread quantization error evenly

---

## EisensteinInteger

### Creation

```python
from snapkit import EisensteinInteger

# Direct construction
ei = EisensteinInteger(3, 1)  # 3 + 1·ω

# From a complex number (rounds to nearest)
ei = EisensteinInteger.from_complex(complex(1.2, 0.7))
# EisensteinInteger(1, 1)
```

### Properties

```python
ei = EisensteinInteger(3, 1)

ei.complex       # complex(-2.5+0.866j) — Cartesian form
ei.norm_squared  # 7 — a² - ab + b² = 9 - 3 + 1
abs(ei)          # 2.646... — sqrt(norm_squared)
```

### Arithmetic

```python
a = EisensteinInteger(3, 1)
b = EisensteinInteger(1, 2)

a + b            # EisensteinInteger(4, 3)
a - b            # EisensteinInteger(2, -1)
a * b            # EisensteinInteger(1, 7)
a.conjugate()    # EisensteinInteger(4, -1)
```

### Snapping

```python
from snapkit import eisenstein_snap, eisenstein_round

# Snap with tolerance check
nearest, distance, is_snap = eisenstein_snap(complex(0.3, 0.7), tolerance=0.5)
# nearest: EisensteinInteger(0, 1)
# distance: 0.1339
# is_snap: True

# Direct round (no tolerance)
ei = eisenstein_round(complex(0.3, 0.7))
```

### Distance

```python
from snapkit import eisenstein_distance

# Lattice distance between two complex points
d = eisenstein_distance(complex(0.3, 0.7), complex(1.2, 0.5))
```

### Fundamental Domain

```python
from snapkit import eisenstein_fundamental_domain

# Reduce to canonical representative (unit rotation + lattice snap)
unit, reduced = eisenstein_fundamental_domain(complex(2.5, 1.3))
```

### Batch Operations

```python
from snapkit import eisenstein_snap_batch

points = [complex(0.3, 0.7), complex(1.1, 0.4), complex(2.5, 1.8)]
results = eisenstein_snap_batch(points, tolerance=0.5)

for nearest, distance, is_snap in results:
    print(f"({nearest.a}, {nearest.b}): dist={distance:.4f}, snap={is_snap}")
```

---

## Voronoï Cell Snap

The Voronoï cell snap finds the **true nearest** Eisenstein integer by checking all 9 candidates in the 3×3 neighborhood.

### Usage

```python
from snapkit.eisenstein_voronoi import (
    eisenstein_snap_voronoi,
    eisenstein_snap_naive,
    eisenstein_to_real,
    snap_distance,
)

# True nearest-neighbor (optimized: squared distance, no sqrt)
a, b = eisenstein_snap_voronoi(0.3, 0.7)
# (0, 1)

# Fast approximate (single rounding)
a, b = eisenstein_snap_naive(0.3, 0.7)
# (0, 1)

# Convert back to Cartesian
x, y = eisenstein_to_real(0, 1)
# (-0.5, 0.866)

# Distance to a specific lattice point
d = snap_distance(0.3, 0.7, 0, 1)
```

### Batch Voronoï

```python
from snapkit.eisenstein_voronoi import eisenstein_snap_batch as voronoi_batch

points = [(0.3, 0.7), (1.1, 0.4), (2.5, 1.8)]
results = voronoi_batch(points)
# [(0, 1), (1, 0), (2, 1)]
```

### Performance Notes

- `eisenstein_snap_voronoi` checks 9 candidates using **squared distance** (no `sqrt` in hot path)
- `eisenstein_snap_naive` is faster but may miss the true nearest in edge cases near Voronoï cell boundaries
- Tie-breaking: prefers smaller `|a|, |b|`

---

## Temporal Snap

### BeatGrid

```python
from snapkit import BeatGrid

grid = BeatGrid(period=1.0, phase=0.0, t_start=0.0)

# Snap a timestamp
result = grid.snap(t=1.04, tolerance=0.1)
result.original_time  # 1.04
result.snapped_time   # 1.0
result.offset         # 0.04
result.is_on_beat     # True (|0.04| ≤ 0.1)
result.beat_index     # 1
result.beat_phase     # 0.04 (position within period)

# Find nearest beat
beat_time, beat_index = grid.nearest_beat(2.7)
# (3.0, 3)

# List beats in range
beats = grid.beats_in_range(0.5, 3.5)
# [1.0, 2.0, 3.0]

# Batch snap
results = grid.snap_batch([0.04, 1.04, 2.51], tolerance=0.1)
```

### TemporalSnap with T-minus-0

T-minus-0 detection finds the moment when a signal crosses zero — the inflection point where derivative changes sign.

```python
from snapkit import TemporalSnap, BeatGrid

grid = BeatGrid(period=1.0)
snap = TemporalSnap(grid, tolerance=0.1, t0_threshold=0.05, t0_window=3)

# Feed observations
snap.observe(t=0.0, value=0.5)
snap.observe(t=1.0, value=0.2)
snap.observe(t=2.0, value=0.001)  # ← near zero, derivative flipped
# result.is_t_minus_0 → True

# Reset
snap.reset()

# Access history
for t, value in snap.history:
    print(f"t={t:.2f}, val={value:.3f}")
```

### Common BeatGrid Patterns

```python
# Musical: 120 BPM, quarter note grid
grid = BeatGrid(period=0.5)  # 0.5s per beat at 120 BPM

# Waltz: every 3 beats
grid = BeatGrid(period=1.5, phase=0.0)

# Syncopated: off-beat grid
grid = BeatGrid(period=1.0, phase=0.5)  # shifted by half
```

---

## Spectral Analysis

### Entropy

```python
from snapkit import entropy

data = [0.1, 0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6]
h = entropy(data, bins=10)
# Returns Shannon entropy in bits
# High entropy → uniform/random distribution
# Low entropy → concentrated/predictable
```

### Hurst Exponent

```python
from snapkit import hurst_exponent

data = [random.gauss(0, 1) for _ in range(500)]
H = hurst_exponent(data)

# H ≈ 0.5 → random walk (no memory)
# H > 0.5 → trending (persistent)
# H < 0.5 → mean-reverting (anti-persistent)
```

### Autocorrelation

```python
from snapkit import autocorrelation

acf = autocorrelation(data, max_lag=50)
# acf[0] = 1.0 (always)
# acf[1] = lag-1 autocorrelation
# acf[k] = lag-k autocorrelation
```

### Full Spectral Summary

```python
from snapkit import spectral_summary

summary = spectral_summary(data, bins=10, max_lag=50)
summary.entropy_bits      # Shannon entropy
summary.hurst             # Hurst exponent
summary.autocorr_lag1     # Lag-1 autocorrelation
summary.autocorr_decay    # Lag where ACF drops below 1/e
summary.is_stationary     # H ∈ [0.4, 0.6] AND |ACF(1)| < 0.3
```

### Batch Analysis

```python
from snapkit import spectral_batch

series_list = [signal_a, signal_b, signal_c]
summaries = spectral_batch(series_list)
for i, s in enumerate(summaries):
    print(f"Series {i}: H={s.hurst:.3f}, stationary={s.is_stationary}")
```

---

## Connectome

### Basic Usage

```python
from snapkit import TemporalConnectome

conn = TemporalConnectome(
    threshold=0.3,    # minimum |correlation| for coupling
    max_lag=5,        # check lags -5 to +5
    min_samples=10,   # minimum data points per pair
)

conn.add_room("alpha", [0.1, 0.5, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.3, 0.5])
conn.add_room("beta",  [0.2, 0.4, 0.4, 0.7, 0.3, 0.6, 0.5, 0.5, 0.4, 0.6])
conn.add_room("gamma", [0.9, 0.1, 0.7, 0.2, 0.8, 0.1, 0.6, 0.3, 0.7, 0.2])

result = conn.analyze()
```

### Analyzing Results

```python
# Coupled pairs (positive correlation)
for pair in result.coupled:
    print(f"{pair.room_a} ↔ {pair.room_b}: r={pair.correlation:.3f}, lag={pair.lag}")

# Anti-coupled pairs (negative correlation)
for pair in result.anti_coupled:
    print(f"{pair.room_a} ↮ {pair.room_b}: r={pair.correlation:.3f}")

# All significant
for pair in result.significant:
    print(f"{pair.coupling.value}: {pair.room_a}, {pair.room_b}")

# Pair properties
pair.room_a        # "alpha"
pair.room_b        # "beta"
pair.coupling      # CouplingType.COUPLED
pair.correlation   # 0.87
pair.lag           # 0 (simultaneous)
pair.confidence    # 0.45
pair.is_significant # True
```

### Adjacency Matrix

```python
names, matrix = result.adjacency_matrix()
# names: ["alpha", "beta", "gamma"]
# matrix: 3x3 correlation matrix (1.0 on diagonal)
```

### Graphviz Export

```python
dot = result.to_graphviz()
print(dot)
# graph Connectome {
#   rankdir=LR;
#   node [shape=circle];
#   "alpha";
#   "beta";
#   "alpha" -- "beta" [color=blue, label="0.87"];
# }
```

---

## FLUX-Tensor-MIDI

### Setting Up

```python
from snapkit import FluxTensorMIDI, TempoMap, MIDIEvent

# Create conductor with tempo map
tempo = TempoMap(ticks_per_beat=480, initial_bpm=120.0)
flux = FluxTensorMIDI(tempo)
```

### Adding Rooms

```python
piano = flux.add_room("piano", channel=0, voice=0)
drums = flux.add_room("drums", channel=9, voice=0)
bass = flux.add_room("bass", channel=1, voice=0)

# Access rooms
flux.room("piano")
flux.rooms  # ["piano", "drums", "bass"]
```

### Scheduling Events

```python
# Note on/off
flux.note_on("piano", tick=0, note=60, velocity=100)    # C4
flux.note_off("piano", tick=480, note=60)                # release after 1 beat
flux.note_on("piano", tick=480, note=64, velocity=80)    # E4
flux.note_off("piano", tick=960, note=64)

flux.note_on("drums", tick=0, note=36, velocity=120)     # kick
flux.note_on("drums", tick=240, note=38, velocity=100)   # snare on beat 2

flux.event_count  # 5
```

### Tempo Changes

```python
tempo.set_tempo(tick=960, bpm=140.0)  # speed up at measure 3
tempo.set_tempo(tick=1920, bpm=100.0) # slow down at measure 5

tempo.bpm_at(480)     # 120.0
tempo.bpm_at(1000)    # 140.0
```

### Tick ↔ Seconds Conversion

```python
tempo.tick_to_seconds(480)   # 1.0 (at 120 BPM, 480 ticks = 1 beat = 0.5s)
tempo.tick_to_seconds(960)   # 2.0 (includes tempo change at 960)
tempo.seconds_to_tick(1.0)   # 480
```

### Rendering

```python
# Render events sorted by tick
events = flux.render()

# Quantize to grid (snap to nearest 16th note)
quantized = flux.quantize(grid=120)  # 120 ticks = 16th note at 480 tpb

# Total duration
duration = flux.timeline_seconds()

# Clear all events
flux.clear()
```

### Working with MIDIEvent

```python
for event in flux.render():
    event.tick        # absolute tick position
    event.channel     # 0-15
    event.event_type  # MIDIEventType.NOTE_ON, NOTE_OFF, etc.
    event.value       # note number or CC number
    event.velocity    # velocity (0 for NOTE_OFF)
```

### Room Helpers

```python
piano = flux.room("piano")

# Room generates events
on_event = piano.note_on(tick=0, note=60, velocity=100)
off_event = piano.note_off(tick=480, note=60)

piano.last_event_tick  # 480
```

---

## Common Patterns

### Quantize a Signal to Eisenstein Lattice

```python
from snapkit import eisenstein_snap

# Snap complex-valued signal samples
signal = [complex(x, y) for x, y in zip(real_part, imag_part)]
quantized = []
for z in signal:
    nearest, dist, is_snap = eisenstein_snap(z, tolerance=0.3)
    quantized.append(nearest.complex)
```

### Detect Onset Times

```python
from snapkit import BeatGrid, TemporalSnap

grid = BeatGrid(period=0.5)  # 120 BPM
snap = TemporalSnap(grid, tolerance=0.05, t0_threshold=0.02)

onsets = []
for t, value in audio_envelope:
    result = snap.observe(t, value)
    if result.is_t_minus_0:
        onsets.append(result.snapped_time)
```

### Analyze Room Coupling in a Musical Ensemble

```python
from snapkit import TemporalConnectome

conn = TemporalConnectome(threshold=0.4, max_lag=10)

# Add activity traces for each musician
for name, notes in ensemble_data.items():
    activity = note_density_over_time(notes)
    conn.add_room(name, activity)

result = conn.analyze()
print(f"Coupled: {len(result.coupled)} pairs")
print(f"Anti-coupled: {len(result.anti_coupled)} pairs")
```

### Build a MID