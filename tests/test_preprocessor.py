"""
Tests for src/preprocessor.py

All tests use synthetic audio (numpy sine waves) so no dataset download
is required to run the test suite.
"""

import numpy as np
import pytest
import sys
import os

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SAMPLE_RATE, MAX_AUDIO_SAMPLES, N_MELS, T_FRAMES, HOP_LENGTH
from src.preprocessor import (
    load_audio,
    pad_or_trim,
    extract_log_mel,
    pad_or_trim_spectrogram,
    normalise_spectrogram,
    preprocess_waveform,
    compute_global_stats,
    set_global_stats,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sine_wave_short():
    """1-second 440 Hz sine wave at 16 kHz."""
    t = np.linspace(0, 1.0, SAMPLE_RATE, dtype=np.float32)
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)


@pytest.fixture
def sine_wave_long():
    """6-second multi-tone sine wave — longer than MAX_AUDIO_SECONDS."""
    t = np.linspace(0, 6.0, 6 * SAMPLE_RATE, dtype=np.float32)
    return (0.3 * np.sin(2 * np.pi * 440 * t) +
            0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)


@pytest.fixture
def audio_file(tmp_path, sine_wave_short):
    """Write a temporary WAV file and return its path."""
    import soundfile as sf
    path = tmp_path / "test_audio.wav"
    sf.write(str(path), sine_wave_short, SAMPLE_RATE, subtype="PCM_16")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# load_audio()
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadAudio:
    def test_returns_1d_array(self, audio_file):
        wav = load_audio(audio_file)
        assert wav.ndim == 1, "Expected 1-D waveform"

    def test_correct_sample_rate(self, audio_file):
        wav = load_audio(audio_file, sr=SAMPLE_RATE)
        # 1-second file → ~16 000 samples
        assert abs(len(wav) - SAMPLE_RATE) < 100, \
            f"Expected ~{SAMPLE_RATE} samples, got {len(wav)}"

    def test_float32_dtype(self, audio_file):
        wav = load_audio(audio_file)
        assert wav.dtype == np.float32

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(RuntimeError):
            load_audio(tmp_path / "nonexistent.wav")


# ─────────────────────────────────────────────────────────────────────────────
# pad_or_trim()
# ─────────────────────────────────────────────────────────────────────────────

class TestPadOrTrim:
    def test_short_waveform_padded(self, sine_wave_short):
        result = pad_or_trim(sine_wave_short, target_length=MAX_AUDIO_SAMPLES)
        assert len(result) == MAX_AUDIO_SAMPLES

    def test_long_waveform_trimmed(self, sine_wave_long):
        result = pad_or_trim(sine_wave_long, target_length=MAX_AUDIO_SAMPLES)
        assert len(result) == MAX_AUDIO_SAMPLES

    def test_exact_length_unchanged(self, sine_wave_short):
        exact = np.zeros(MAX_AUDIO_SAMPLES, dtype=np.float32)
        exact[:len(sine_wave_short)] = sine_wave_short
        result = pad_or_trim(exact, target_length=MAX_AUDIO_SAMPLES)
        assert len(result) == MAX_AUDIO_SAMPLES

    def test_padding_is_zeros(self, sine_wave_short):
        result = pad_or_trim(sine_wave_short, target_length=MAX_AUDIO_SAMPLES)
        # The padded portion should be all zeros
        pad_portion = result[len(sine_wave_short):]
        assert np.all(pad_portion == 0.0)

    def test_trimming_preserves_start(self, sine_wave_long):
        result = pad_or_trim(sine_wave_long, target_length=MAX_AUDIO_SAMPLES)
        np.testing.assert_array_equal(result, sine_wave_long[:MAX_AUDIO_SAMPLES])


# ─────────────────────────────────────────────────────────────────────────────
# extract_log_mel()
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractLogMel:
    def test_output_shape(self, sine_wave_short):
        wav    = pad_or_trim(sine_wave_short, MAX_AUDIO_SAMPLES)
        spec   = extract_log_mel(wav)
        h, w   = spec.shape
        assert h == N_MELS, f"Expected {N_MELS} mel bins, got {h}"
        assert w == T_FRAMES, f"Expected ~{T_FRAMES} frames, got {w}"

    def test_output_dtype(self, sine_wave_short):
        wav  = pad_or_trim(sine_wave_short, MAX_AUDIO_SAMPLES)
        spec = extract_log_mel(wav)
        assert spec.dtype == np.float32

    def test_dB_range(self, sine_wave_short):
        """Log-Mel values should be in dB range (negative, up to ~0 dB)."""
        wav  = pad_or_trim(sine_wave_short, MAX_AUDIO_SAMPLES)
        spec = extract_log_mel(wav)
        assert spec.max() <= 1.0, f"Max dB too high: {spec.max()}"
        assert spec.min() < 0.0, "Expected negative dB values"

    def test_silent_audio(self):
        silence = np.zeros(MAX_AUDIO_SAMPLES, dtype=np.float32)
        spec    = extract_log_mel(silence)
        assert spec.shape == (N_MELS, T_FRAMES)
        # All values should be very negative (silent = -80 dB) or 0.0 if not ref
        assert spec.max() <= 0.0, "Silent audio should not produce positive dB values"


# ─────────────────────────────────────────────────────────────────────────────
# pad_or_trim_spectrogram()
# ─────────────────────────────────────────────────────────────────────────────

class TestPadOrTrimSpectrogram:
    def test_short_spec_padded(self):
        spec   = np.ones((N_MELS, 100), dtype=np.float32)
        result = pad_or_trim_spectrogram(spec, target_frames=T_FRAMES)
        assert result.shape == (N_MELS, T_FRAMES)

    def test_long_spec_trimmed(self):
        spec   = np.ones((N_MELS, T_FRAMES + 200), dtype=np.float32)
        result = pad_or_trim_spectrogram(spec, target_frames=T_FRAMES)
        assert result.shape == (N_MELS, T_FRAMES)

    def test_exact_spec_unchanged_shape(self):
        spec   = np.random.randn(N_MELS, T_FRAMES).astype(np.float32)
        result = pad_or_trim_spectrogram(spec, target_frames=T_FRAMES)
        assert result.shape == (N_MELS, T_FRAMES)


# ─────────────────────────────────────────────────────────────────────────────
# normalise_spectrogram()
# ─────────────────────────────────────────────────────────────────────────────

class TestNormaliseSpectrogram:
    def test_shape_preserved(self):
        spec   = np.random.randn(N_MELS, T_FRAMES).astype(np.float32) * 20 - 40
        result = normalise_spectrogram(spec, mean=-40.0, std=20.0)
        assert result.shape == spec.shape

    def test_zero_mean_unit_std_approx(self):
        """After normalising with the correct stats, the output should be ~N(0,1)."""
        rng  = np.random.default_rng(0)
        spec = (rng.standard_normal((N_MELS, T_FRAMES)) * 20 - 40).astype(np.float32)
        result = normalise_spectrogram(spec, mean=-40.0, std=20.0)
        assert abs(float(result.mean())) < 0.5
        assert 0.8 < float(result.std()) < 1.2

    def test_dtype_float32(self):
        spec   = np.ones((N_MELS, T_FRAMES), dtype=np.float64)
        result = normalise_spectrogram(spec)
        assert result.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# preprocess_waveform()  — full pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessWaveform:
    def test_output_shape_with_channel(self, sine_wave_short):
        result = preprocess_waveform(sine_wave_short, add_channel_dim=True)
        assert result.shape == (N_MELS, T_FRAMES, 1), \
            f"Expected ({N_MELS}, {T_FRAMES}, 1), got {result.shape}"

    def test_output_shape_without_channel(self, sine_wave_short):
        result = preprocess_waveform(sine_wave_short, add_channel_dim=False)
        assert result.shape == (N_MELS, T_FRAMES)

    def test_identical_for_same_input(self, sine_wave_short):
        """Deterministic: same input → same output."""
        r1 = preprocess_waveform(sine_wave_short)
        r2 = preprocess_waveform(sine_wave_short)
        np.testing.assert_array_equal(r1, r2)

    def test_long_audio_trimmed(self, sine_wave_long):
        result = preprocess_waveform(sine_wave_long)
        assert result.shape[0] == N_MELS
        assert result.shape[1] == T_FRAMES


# ─────────────────────────────────────────────────────────────────────────────
# compute_global_stats() / set_global_stats()
# ─────────────────────────────────────────────────────────────────────────────

class TestGlobalStats:
    def test_compute_returns_floats(self, tmp_path):
        import soundfile as sf
        paths = []
        for i in range(5):
            wav  = (np.sin(np.linspace(0, 2 * np.pi * (440 + i * 50), SAMPLE_RATE))
                    * 0.5).astype(np.float32)
            path = tmp_path / f"tone_{i}.wav"
            sf.write(str(path), wav, SAMPLE_RATE)
            paths.append(str(path))

        mean, std = compute_global_stats(paths)
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert std > 0, "std should be positive"

    def test_set_global_stats_updates_module(self):
        from src import preprocessor
        original_mean = preprocessor._GLOBAL_MEAN
        set_global_stats(mean=-30.0, std=15.0)
        assert preprocessor._GLOBAL_MEAN == -30.0
        assert preprocessor._GLOBAL_STD  == 15.0
        # Restore
        set_global_stats(mean=original_mean, std=20.0)
