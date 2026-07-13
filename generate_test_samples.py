"""
Generate diverse test audio samples for DeepShield Audio app testing.
Produces 8 WAV files covering a wide range of acoustic properties:
  1. pure_sine_440hz      — clean single-tone sine wave
  2. chord_harmonics      — rich harmonic chord (vocal-like formants)
  3. white_noise          — broadband noise
  4. pink_noise           — 1/f noise (more speech-like spectrum)
  5. voiced_speech_sim    — amplitude-modulated voiced speech simulation
  6. whisper_sim          — low-amplitude breathy noise burst
  7. silence_with_clicks  — near-silence with artifact clicks
  8. long_audio_6s        — 6-second clip for windowed analysis testing
"""

import os
import numpy as np
import wave
import struct

SR   = 16000
DTYPE = np.float32

def _save_wav(filename: str, waveform: np.ndarray, sr: int = SR):
    """Save a float32 waveform as 16-bit PCM WAV."""
    # Normalise to avoid clipping
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak * 0.9
    pcm = (waveform * 32767).astype(np.int16)
    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))
    print(f"  ✅ {filename}  ({len(waveform)/sr:.2f} s)")

def make_sine(freq=440, duration=4.0, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(DTYPE)

def make_chord(freqs=(261.63, 329.63, 392.0), duration=4.0, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t)
    for i, f in enumerate(freqs):
        # Decaying harmonics for vocal-like quality
        for h in range(1, 6):
            signal += (1.0 / h) * np.sin(2 * np.pi * f * h * t)
    return signal.astype(DTYPE)

def make_white_noise(duration=4.0, sr=SR):
    rng = np.random.default_rng(42)
    return rng.standard_normal(int(sr * duration)).astype(DTYPE)

def make_pink_noise(duration=4.0, sr=SR):
    """Voss-McCartney pink noise (1/f)."""
    n = int(sr * duration)
    rng = np.random.default_rng(7)
    # Simple approximation: sum of octave-band noise
    num_cols = 16
    array = rng.standard_normal((n, num_cols))
    array = np.cumsum(array, axis=0)
    array /= (np.arange(num_cols) + 1)
    pink = np.sum(array, axis=1)
    return pink.astype(DTYPE)

def make_voiced_speech_sim(duration=4.0, sr=SR):
    """Voiced speech: glottal pulses at ~120 Hz, shaped by vocal-tract formants."""
    t  = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Glottal source: periodic pulse train
    f0 = 120.0
    source = np.sin(2 * np.pi * f0 * t) ** 3
    # Formant filter approximation: sum of resonances
    formants = [(700, 80), (1220, 90), (2600, 120), (3400, 150)]
    signal = np.zeros_like(t)
    for fc, bw in formants:
        # Simple resonance = sine at formant freq, amplitude-weighted
        signal += (fc / 700) ** -0.5 * np.sin(2 * np.pi * fc * t) * source
    # Add natural amplitude envelope (syllable-like)
    env = 0.5 * (1 + np.sin(2 * np.pi * 3.0 * t - np.pi / 2))  # ~3 Hz modulation
    return (signal * env).astype(DTYPE)

def make_whisper_sim(duration=4.0, sr=SR):
    """Whispered speech: breathy, turbulent noise modulated by mouth shape."""
    rng = np.random.default_rng(99)
    n   = int(sr * duration)
    noise = rng.standard_normal(n)
    # Low-pass at 4 kHz by simple FIR
    from numpy.fft import rfft, irfft
    F = rfft(noise)
    freqs = np.arange(len(F)) * sr / (2 * len(F))
    cutoff = 4000
    F[freqs > cutoff] *= 0.05
    signal = irfft(F, n=n)
    # Syllabic amplitude envelope
    t   = np.linspace(0, duration, n)
    env = np.clip(0.7 + 0.3 * np.sin(2 * np.pi * 4.0 * t), 0, 1)
    return (signal * env * 0.25).astype(DTYPE)

def make_silence_with_clicks(duration=4.0, sr=SR):
    """Near-silence with occasional click artifacts (quantisation noise test)."""
    n = int(sr * duration)
    signal = np.zeros(n, dtype=DTYPE)
    rng = np.random.default_rng(13)
    # Sparse random clicks
    for _ in range(12):
        pos = rng.integers(0, n)
        signal[pos] = rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 0.9)
    # Very low level broadband noise floor
    signal += rng.standard_normal(n).astype(DTYPE) * 0.002
    return signal

def make_long_audio(duration=6.5, sr=SR):
    """6.5-second audio: voiced section + noise burst + silence — tests windowed mode."""
    n        = int(sr * duration)
    signal   = np.zeros(n, dtype=DTYPE)
    voiced   = make_voiced_speech_sim(duration=3.0, sr=sr)
    noise    = make_white_noise(duration=2.0, sr=sr) * 0.4
    whisper  = make_whisper_sim(duration=1.5, sr=sr)
    t0 = 0
    signal[t0 : t0 + len(voiced)] += voiced
    t0 += len(voiced)
    signal[t0 : t0 + len(noise)] += noise
    t0 += len(noise)
    signal[t0 : t0 + len(whisper)] += whisper
    return signal


if __name__ == "__main__":
    out_dir = "test_samples"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n🎵 Generating test audio samples → {out_dir}/\n")

    _save_wav(f"{out_dir}/01_pure_sine_440hz.wav",       make_sine(440))
    _save_wav(f"{out_dir}/02_chord_harmonics.wav",       make_chord())
    _save_wav(f"{out_dir}/03_white_noise.wav",           make_white_noise())
    _save_wav(f"{out_dir}/04_pink_noise.wav",            make_pink_noise())
    _save_wav(f"{out_dir}/05_voiced_speech_sim.wav",     make_voiced_speech_sim())
    _save_wav(f"{out_dir}/06_whisper_sim.wav",           make_whisper_sim())
    _save_wav(f"{out_dir}/07_silence_with_clicks.wav",   make_silence_with_clicks())
    _save_wav(f"{out_dir}/08_long_audio_6s.wav",         make_long_audio())

    print(f"\n✅ All 8 samples saved to ./{out_dir}/")
    print("   Upload them one by one in the DeepShield Audio app at http://localhost:8501")
    print("   Use sample 08 to test Windowed Analysis mode.\n")
