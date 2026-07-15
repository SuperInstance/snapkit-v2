"""Tests for audio synthesis."""

import math
import pytest
from snapkit.audio import (
    midi_to_freq, write_wav, synthesize_events, synthesize_demo,
    harmony_demo_audio,
)


class TestMidiToFreq:
    def test_middle_a(self):
        # A4 (note 69) = 440 Hz
        assert abs(midi_to_freq(69) - 440.0) < 0.1

    def test_octave_doubling(self):
        # Each octave doubles frequency
        assert abs(midi_to_freq(81) / midi_to_freq(69) - 2.0) < 0.01

    def test_low_note(self):
        # A0 (note 21) = 27.5 Hz
        assert abs(midi_to_freq(21) - 27.5) < 0.1


class TestWriteWav:
    def test_write_silence(self, tmp_path):
        filename = str(tmp_path / "test.wav")
        write_wav([0.0] * 100, filename)
        import os
        assert os.path.exists(filename)
        assert os.path.getsize(filename) > 0

    def test_clip_samples(self, tmp_path):
        filename = str(tmp_path / "clip.wav")
        # Values > 1.0 should be clipped
        write_wav([2.0, -2.0, 0.5], filename)
        assert True  # Just verify it doesn't crash


class TestSynthesizeDemo:
    def test_produces_samples(self):
        samples = synthesize_demo(duration_seconds=5.0)
        assert len(samples) > 0
        assert all(-1.0 <= s <= 1.0 for s in samples)

    def test_harmony_demo_audio(self, tmp_path, monkeypatch):
        # Use a temp file
        filename = str(tmp_path / "harmony.wav")
        harmony_demo_audio(filename=filename)
        import os
        assert os.path.exists(filename)
        # WAV file should be at least 100KB for 20s audio
        assert os.path.getsize(filename) > 100_000


class TestSynthesizeEvents:
    def test_empty_events(self):
        samples = synthesize_events([], duration_seconds=1.0)
        assert len(samples) == 22050
        assert all(s == 0.0 for s in samples)