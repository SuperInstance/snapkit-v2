"""Tests for v2.8 fixes from MiniMax-M3 review."""
import math
import time
import pytest
from snapkit.spectral import autocorrelation
from snapkit.midi_io import TempoDeriver
from snapkit.midi import FluxTensorMIDI
from snapkit.fleet import FleetCoordinator, VesselSnapshot, FleetAlert
from snapkit.eisenstein import eisenstein_distance
from snapkit.clever_tokens import TokenLattice
from snapkit.connectome import TemporalConnectome
from snapkit.audio import synthesize_demo


class TestAutocorrelationOverflow:
    def test_huge_values_no_overflow(self):
        data = [1e200 + (i * 1e10) for i in range(64)]
        acf = autocorrelation(data, max_lag=10)
        assert acf[0] == 1.0
        for i in range(1, len(acf)):
            assert acf[i] == acf[i], f"NaN at lag {i}"
            assert acf[i] != float('inf'), f"inf at lag {i}"

    def test_normal_data_unchanged(self):
        data = [math.sin(i * 0.1) + 0.1 for i in range(64)]
        acf = autocorrelation(data, max_lag=5)
        assert acf[0] == pytest.approx(1.0, abs=1e-6)


class TestTempoDeriverSampleRate:
    def test_default_is_10hz(self):
        d = TempoDeriver()
        assert d._sample_rate_hz == 10.0

    def test_custom_rate(self):
        d = TempoDeriver(sample_rate_hz=100.0)
        assert d._sample_rate_hz == 100.0


class TestFluxTensorMIDIRender:
    def test_render_returns_copy(self):
        bus = FluxTensorMIDI()
        bus.add_room("test", channel=0)
        bus.note_on("test", 0, 60, velocity=100)
        bus.note_on("test", 10, 64, velocity=100)
        r1 = bus.render()
        n = len(r1)
        r1.clear()
        r2 = bus.render()
        assert len(r2) == n, f"Render: {len(r2)} vs original {n}"


class TestFleetDriftThreshold:
    def _snap(self, vid, heading, now):
        return VesselSnapshot(
            vessel_id=vid, channel=0, phi=0.5, bpm=60,
            heading=heading, heading_error=0, sea_state="rough",
            is_in_harmony=True, is_strained=False, is_surprised=False,
            last_update=now, alarms=0,
        )

    def test_threshold_respected(self):
        now = time.time()
        fc = FleetCoordinator(drift_threshold=20.0)
        fc.register_vessel("a", 0)
        fc.register_vessel("b", 1)
        fc.update_vessel(self._snap("a", 0, now))
        fc.update_vessel(self._snap("b", 10, now))
        fc.detect_events()
        assert not any(e.alert_type == FleetAlert.DRIFT for e in fc.events)

    def test_threshold_triggers(self):
        now = time.time()
        fc = FleetCoordinator(drift_threshold=15.0)
        fc.register_vessel("a", 0)
        fc.register_vessel("b", 1)
        fc.update_vessel(self._snap("a", 0, now))
        fc.update_vessel(self._snap("b", 50, now))
        fc.detect_events()
        assert any(e.alert_type == FleetAlert.DRIFT for e in fc.events)


class TestEisensteinDistance:
    def test_triangle_inequality(self):
        pts = [complex(0,0), complex(3,4), complex(-2,7), complex(11,-5)]
        for a in pts:
            for b in pts:
                for c in pts:
                    ab = eisenstein_distance(a, b)
                    bc = eisenstein_distance(b, c)
                    ac = eisenstein_distance(a, c)
                    assert ac <= ab + bc + 1e-9

    def test_symmetry(self):
        assert eisenstein_distance(1+2j, 3+5j) == pytest.approx(
            eisenstein_distance(3+5j, 1+2j))

    def test_zero(self):
        assert eisenstein_distance(1+1j, 1+1j) == 0.0


class TestLatticeDistances:
    def test_euclidean(self):
        l = TokenLattice()
        from snapkit.clever_tokens import ConstraintType
        l.register_token("a", lattice_coord=(0,0))
        l.register_token("b", lattice_coord=(1,2))
        d = l.lattice_distances()
        assert len(d) >= 1
        for v in d.values():
            assert v >= 0


class TestReplaceTraces:
    def test_swap_atomically(self):
        tc = TemporalConnectome()
        tc.replace_traces({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        })
        result = tc.analyze()
        assert result.pairs[0].coupling.value == "coupled"


class TestSynthesizeDemoReachesDissonance:
    def test_dissonance_plays(self):
        s = synthesize_demo(duration_seconds=20.0)
        assert len(s) == 20 * 22050
        dw = s[int(8*22050):int(12*22050)]
        nz = sum(1 for x in dw if abs(x) > 0.001)
        assert nz > 100, f"Dissonance silent: only {nz} non-zero"
