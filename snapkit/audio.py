"""
Audio Synthesis — Listen to the harmony.

Takes MIDI events from the FluxTensorMIDI bus and synthesizes them as audio
so you can HEAR the harmony shatter and recover. This is the visceral proof
that the architecture works — when something goes wrong, you hear dissonance.

Each channel becomes a voice. The velocity becomes volume. The tempo
becomes the beat. When the Governor detects friction, the harmony breaks.

Uses only stdlib + the built-in `struct` module to write raw WAV files.
No external audio dependencies.

Usage:
    python3 -m snapkit.listen harmony     # Play the demo as audio
    python3 -m snapkit.listen fleet      # Play fleet simulation
    python3 -m snapkit.listen file.mid   # Play any MIDI file
"""

import math
import struct
import wave
from typing import List, Optional

from snapkit.midi import MIDIEvent, MIDIEventType


def midi_to_freq(note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2 ** ((note - 69) / 12))


def write_wav(
    samples: List[float],
    filename: str,
    sample_rate: int = 22050,
) -> None:
    """Write a list of float samples (-1.0 to 1.0) to a WAV file."""
    with wave.open(filename, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        for s in samples:
            clipped = max(-1.0, min(1.0, s))
            wav.writeframes(struct.pack('<h', int(clipped * 32767)))


def synthesize_events(
    events: List[MIDIEvent],
    duration_seconds: float = 10.0,
    sample_rate: int = 22050,
    bpm: float = 120.0,
    beats_per_tick: float = 1.0,
    waveform: str = 'sine',
) -> List[float]:
    """Synthesize MIDI events as audio samples.

    Args:
        events: Sorted MIDI events (from FluxTensorMIDI.render()).
        duration_seconds: Length of output audio.
        sample_rate: Audio sample rate.
        bpm: Tempo for translating ticks to time.
        waveform: 'sine', 'square', 'saw', 'triangle'.
    """
    seconds_per_beat = 60.0 / bpm
    ticks_per_beat = 480  # Default MIDI ticks per beat
    seconds_per_tick = seconds_per_beat / ticks_per_beat * beats_per_tick

    total_samples = int(duration_seconds * sample_rate)
    samples = [0.0] * total_samples

    # Group events into notes (note_on + matching note_off)
    active_notes = []  # (channel, note, start_sample, velocity)
    completed_notes = []  # (channel, note, start_sample, end_sample, velocity)

    last_tick = 0
    for event in events:
        tick_delta = event.tick - last_tick
        last_tick = event.tick
        sample_delta = int(tick_delta * seconds_per_tick * sample_rate)

        # Render active notes (shift their start time by the delta so they
        # advance to the right sample position).
        for note in active_notes:
            note[2] += sample_delta

        # Convert event to absolute sample position from its tick.
        event_pos = int(event.tick * seconds_per_tick * sample_rate)

        if event.event_type == MIDIEventType.NOTE_ON:
            active_notes.append([
                event.channel, event.value, event_pos, event.velocity,
            ])
        elif event.event_type == MIDIEventType.NOTE_OFF:
            for note in active_notes:
                if note[1] == event.value and note[0] == event.channel:
                    # NOTE_ON start was recorded at note[2]; NOTE_OFF is at
                    # event_pos. start < end is enforced.
                    start = note[2]
                    end = max(start + 1, event_pos)
                    completed_notes.append(
                        (note[0], note[1], start, end, note[3])
                    )
                    active_notes.remove(note)
                    break

    # Synthesize each completed note
    for channel, note_num, start, end, velocity in completed_notes:
        if end <= start:
            end = start + int(0.1 * sample_rate)
        freq = midi_to_freq(note_num)
        amp = velocity / 127.0 * 0.3  # Scale to prevent clipping

        # Add slight detune per channel for thickness
        detune = channel * 0.005
        freq *= (1 + detune)

        for i in range(max(0, start), min(end, len(samples))):
            t = (i - start) / sample_rate
            # ADSR envelope
            attack = 0.02
            release = 0.1
            if t < attack:
                env = t / attack
            elif t > (end - start) / sample_rate - release:
                env = max(0, ((end - start) / sample_rate - t) / release)
            else:
                env = 1.0

            phase = 2 * math.pi * freq * t
            if waveform == 'sine':
                wave_val = math.sin(phase)
            elif waveform == 'square':
                wave_val = 1.0 if math.sin(phase) > 0 else -1.0
            elif waveform == 'saw':
                wave_val = 2 * (t * freq - math.floor(0.5 + t * freq))
            elif waveform == 'triangle':
                wave_val = 2 * abs(2 * (t * freq - math.floor(0.5 + t * freq))) - 1
            else:
                wave_val = math.sin(phase)

            # Soft saturation
            samples[i] += wave_val * amp * env * 0.5

    # Soft clip to prevent clipping
    for i in range(len(samples)):
        samples[i] = math.tanh(samples[i])

    return samples


def synthesize_demo(duration_seconds: float = 30.0) -> List[float]:
    """Generate a demo harmony sequence for audio playback.

    Plays a melody in C major, then introduces dissonance, then resolves.
    """
    sample_rate = 22050
    total_samples = int(duration_seconds * sample_rate)
    samples = [0.0] * total_samples

    # Chord progression arc — 20 seconds, 10 chords at 2s each.
    # 0:00-8:00 — harmony (C - Am - F - G twice)
    # 8:00-12:00 — dissonance (sharp clusters)
    # 12:00-20:00 — resolution back to C
    #
    # Each tuple is (chord_notes, is_dissonant, description).
    # The dissonance section must actually be reached — this was a P2 bug
    # in the original: only 4 chords in the array, so chord_idx >= 4 was
    # unreachable code.
    progression = [
        ([60, 64, 67], False, "C major — harmony"),       # 0
        ([57, 60, 64], False, "A minor — harmony"),       # 1
        ([53, 57, 60], False, "F major — harmony"),       # 2
        ([55, 59, 62], False, "G major — harmony"),       # 3
        ([60, 64, 67], False, "C major — harmony"),       # 4 (repeat to fill harmony section)
        ([61, 64, 67, 70], True, "Cluster — dissonance"), # 5
        ([58, 63, 66, 71], True, "More dissonance"),      # 6
        ([55, 59, 60, 64], False, "Resolving"),           # 7
        ([60, 64, 67], False, "C major — resolution"),    # 8
        ([60, 64, 67, 72], False, "C with octave — home"),# 9
    ]

    bpm = 120
    seconds_per_chord = 2.0  # Each chord lasts 2 seconds

    for chord_idx, (chord_notes, is_dissonant, _desc) in enumerate(progression):
        start = int(chord_idx * seconds_per_chord * sample_rate)
        end = int((chord_idx + 1) * seconds_per_chord * sample_rate)
        if start >= total_samples:
            break

        for note_num in chord_notes:
            freq = midi_to_freq(note_num)
            amp = 0.15
            for i in range(start, min(end, total_samples)):
                t = (i - start) / sample_rate
                # ADSR
                attack = 0.05
                release = 0.3
                dur = (end - start) / sample_rate
                if t < attack:
                    env = t / attack
                elif t > dur - release:
                    env = max(0, (dur - t) / release)
                else:
                    env = 1.0

                phase = 2 * math.pi * freq * t
                wave_val = math.sin(phase) * env * amp
                # Add slight detune for thickness
                wave_val += math.sin(phase * 1.003) * env * amp * 0.5
                samples[i] += wave_val

    # Soft clip
    for i in range(len(samples)):
        samples[i] = math.tanh(samples[i])

    return samples


def harmony_demo_audio(filename: str = "/tmp/harmony_demo.wav") -> None:
    """Generate and save the harmony demo as audio."""
    print(f"Synthesizing harmony demo → {filename}")
    samples = synthesize_demo(duration_seconds=20.0)
    write_wav(samples, filename)
    print(f"  Saved {len(samples)} samples")
    print(f"  Duration: {len(samples) / 22050:.1f}s")
    print()
    print("  Listen for:")
    print("    0:00-8:00 — Harmony (C - Am - F - G progression)")
    print("    8:00-12:00 — Dissonance (sharp clusters)")
    print("    12:00-20:00 — Resolution back to C")
    print()
    print("  This is what the Executive hears when a model breaks.")