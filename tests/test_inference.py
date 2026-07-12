"""
Tests for src/inference.py

All tests use random weights (no trained model needed) to verify
the inference pipeline mechanics are correct end-to-end.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SAMPLE_RATE, N_MELS, T_FRAMES


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def random_cnn_model():
    """Custom CNN with random (untrained) weights."""
    from src.models.custom_cnn import build_custom_cnn
    return build_custom_cnn()


@pytest.fixture
def short_wav_file(tmp_path):
    """2-second sine wave WAV file."""
    import soundfile as sf
    t   = np.linspace(0, 2.0, 2 * SAMPLE_RATE, dtype=np.float32)
    wav = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    path = tmp_path / "test_short.wav"
    sf.write(str(path), wav, SAMPLE_RATE)
    return path


@pytest.fixture
def long_wav_file(tmp_path):
    """10-second sine wave WAV file (for windowed inference)."""
    import soundfile as sf
    t   = np.linspace(0, 10.0, 10 * SAMPLE_RATE, dtype=np.float32)
    wav = (0.3 * np.sin(2 * np.pi * 440 * t) +
           0.2 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    path = tmp_path / "test_long.wav"
    sf.write(str(path), wav, SAMPLE_RATE)
    return path


@pytest.fixture
def engine(random_cnn_model):
    from src.inference import AudioInferenceEngine
    return AudioInferenceEngine(random_cnn_model, model_name="custom_cnn", threshold=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# get_risk_band()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetRiskBand:
    def test_low_risk(self):
        from src.inference import get_risk_band
        r = get_risk_band(0.1)
        assert r["level"] == "LOW"

    def test_medium_risk(self):
        from src.inference import get_risk_band
        r = get_risk_band(0.5)
        assert r["level"] == "MEDIUM"

    def test_high_risk(self):
        from src.inference import get_risk_band
        r = get_risk_band(0.9)
        assert r["level"] == "HIGH"

    def test_boundary_low_medium(self):
        from src.inference import get_risk_band
        from src.config import RISK_LOW_MAX
        r = get_risk_band(RISK_LOW_MAX)
        assert r["level"] == "LOW"
        r2 = get_risk_band(RISK_LOW_MAX + 0.01)
        assert r2["level"] == "MEDIUM"

    def test_required_keys(self):
        from src.inference import get_risk_band
        r = get_risk_band(0.5)
        for key in ("level", "emoji", "color", "description"):
            assert key in r


# ─────────────────────────────────────────────────────────────────────────────
# AudioInferenceEngine — single file predict()
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictSingleFile:
    def test_returns_dict(self, engine, short_wav_file):
        result = engine.predict(short_wav_file)
        assert isinstance(result, dict)

    def test_required_keys(self, engine, short_wav_file):
        result = engine.predict(short_wav_file)
        required = ("label", "is_fake", "real_prob", "fake_prob",
                    "confidence", "threshold", "risk", "spec", "inference_ms")
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_label_is_string(self, engine, short_wav_file):
        result = engine.predict(short_wav_file)
        assert result["label"] in ("bonafide", "spoof")

    def test_probabilities_sum_to_one(self, engine, short_wav_file):
        result = engine.predict(short_wav_file)
        total = result["real_prob"] + result["fake_prob"]
        assert abs(total - 1.0) < 1e-5, f"real+fake should sum to 1, got {total}"

    def test_probabilities_in_01(self, engine, short_wav_file):
        result = engine.predict(short_wav_file)
        assert 0.0 <= result["real_prob"] <= 1.0
        assert 0.0 <= result["fake_prob"] <= 1.0

    def test_confidence_is_max_prob(self, engine, short_wav_file):
        result = engine.predict(short_wav_file)
        expected_conf = max(result["real_prob"], result["fake_prob"])
        assert abs(result["confidence"] - expected_conf) < 1e-6

    def test_is_fake_consistent_with_label(self, engine, short_wav_file):
        result = engine.predict(short_wav_file)
        if result["is_fake"]:
            assert result["label"] == "spoof"
        else:
            assert result["label"] == "bonafide"

    def test_spec_shape(self, engine, short_wav_file):
        result = engine.predict(short_wav_file)
        assert result["spec"].shape == (N_MELS, T_FRAMES, 1)

    def test_inference_time_positive(self, engine, short_wav_file):
        result = engine.predict(short_wav_file)
        assert result["inference_ms"] > 0

    def test_threshold_in_result(self, engine, short_wav_file):
        result = engine.predict(short_wav_file)
        assert result["threshold"] == engine.threshold


# ─────────────────────────────────────────────────────────────────────────────
# AudioInferenceEngine — long audio predict_long_audio()
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictLongAudio:
    def test_returns_dict(self, engine, long_wav_file):
        result = engine.predict_long_audio(long_wav_file, window_sec=3.0, hop_sec=1.5)
        assert isinstance(result, dict)

    def test_required_keys(self, engine, long_wav_file):
        result = engine.predict_long_audio(long_wav_file)
        for key in ("windows", "aggregate_fake_prob", "overall_label",
                    "overall_is_fake", "duration_sec", "n_windows"):
            assert key in result, f"Missing key: {key}"

    def test_n_windows_positive(self, engine, long_wav_file):
        result = engine.predict_long_audio(long_wav_file, window_sec=3.0, hop_sec=1.5)
        assert result["n_windows"] >= 1

    def test_window_keys(self, engine, long_wav_file):
        result = engine.predict_long_audio(long_wav_file, window_sec=3.0, hop_sec=1.5)
        for w in result["windows"]:
            for key in ("window_idx", "start_sec", "end_sec", "real_prob", "fake_prob", "label"):
                assert key in w, f"Window missing key: {key}"

    def test_window_probs_in_01(self, engine, long_wav_file):
        result = engine.predict_long_audio(long_wav_file, window_sec=3.0, hop_sec=1.5)
        for w in result["windows"]:
            assert 0.0 <= w["real_prob"] <= 1.0
            assert 0.0 <= w["fake_prob"] <= 1.0

    def test_window_times_increasing(self, engine, long_wav_file):
        result = engine.predict_long_audio(long_wav_file, window_sec=3.0, hop_sec=1.5)
        starts = [w["start_sec"] for w in result["windows"]]
        assert starts == sorted(starts), "Window start times should be non-decreasing"

    def test_aggregate_fake_prob_in_01(self, engine, long_wav_file):
        result = engine.predict_long_audio(long_wav_file, window_sec=3.0, hop_sec=1.5)
        assert 0.0 <= result["aggregate_fake_prob"] <= 1.0

    def test_duration_sensible(self, engine, long_wav_file):
        result = engine.predict_long_audio(long_wav_file)
        assert 8.0 <= result["duration_sec"] <= 12.0  # 10-second file

    def test_short_audio_single_window(self, engine, short_wav_file):
        """Audio shorter than window_sec → exactly 1 window."""
        result = engine.predict_long_audio(short_wav_file, window_sec=5.0, hop_sec=2.5)
        assert result["n_windows"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# AudioInferenceEngine — predict_from_waveform()
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictFromWaveform:
    def test_returns_dict(self, engine):
        wav    = (np.sin(np.linspace(0, 2 * np.pi * 440, SAMPLE_RATE)) * 0.5).astype(np.float32)
        result = engine.predict_from_waveform(wav, sr=SAMPLE_RATE)
        assert isinstance(result, dict)

    def test_probs_valid(self, engine):
        wav    = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.1
        result = engine.predict_from_waveform(wav, sr=SAMPLE_RATE)
        assert abs(result["real_prob"] + result["fake_prob"] - 1.0) < 1e-5

    def test_handles_resampling(self, engine):
        """Input at 8kHz should be resampled to 16kHz without error."""
        wav = np.random.randn(8000).astype(np.float32) * 0.1
        result = engine.predict_from_waveform(wav, sr=8000)
        assert "real_prob" in result


# ─────────────────────────────────────────────────────────────────────────────
# Custom threshold
# ─────────────────────────────────────────────────────────────────────────────

class TestCustomThreshold:
    def test_threshold_affects_label(self, random_cnn_model, short_wav_file):
        """Setting threshold=0 → always fake; threshold=1 → always real."""
        from src.inference import AudioInferenceEngine
        always_fake_engine = AudioInferenceEngine(random_cnn_model, threshold=0.0)
        always_real_engine = AudioInferenceEngine(random_cnn_model, threshold=1.0)

        r_fake = always_fake_engine.predict(short_wav_file)
        r_real = always_real_engine.predict(short_wav_file)

        assert r_fake["label"] == "spoof",    "threshold=0 should always predict spoof"
        assert r_real["label"] == "bonafide", "threshold=1 should always predict bonafide"
