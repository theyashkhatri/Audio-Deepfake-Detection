"""
DeepShield Audio — Inference Engine
======================================
Production-ready inference API for both single-file and long-audio
(windowed) prediction.

All preprocessing is done via the shared preprocessor.py module —
identical to training-time — preventing any preprocessing drift.

Usage:
    from src.inference import AudioInferenceEngine
    engine = AudioInferenceEngine(model, model_name="custom_cnn")
    result = engine.predict("audio.wav")
    windowed = engine.predict_long("long_audio.wav")
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

try:
    from src.config import (
        SAMPLE_RATE, MAX_AUDIO_SAMPLES, T_FRAMES, N_MELS,
        WINDOW_SECONDS, HOP_SECONDS, DEFAULT_THRESHOLD,
        RISK_LOW_MAX, RISK_MED_MAX, EFFICIENTNET_INPUT_SIZE,
    )
    from src.preprocessor import (
        load_audio, preprocess_file, preprocess_waveform,
    )
    from src.threshold_optimizer import load_optimal_threshold
except ImportError:
    from config import (
        SAMPLE_RATE, MAX_AUDIO_SAMPLES, T_FRAMES, N_MELS,
        WINDOW_SECONDS, HOP_SECONDS, DEFAULT_THRESHOLD,
        RISK_LOW_MAX, RISK_MED_MAX, EFFICIENTNET_INPUT_SIZE,
    )
    from preprocessor import (
        load_audio, preprocess_file, preprocess_waveform,
    )
    from threshold_optimizer import load_optimal_threshold

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RISK BAND
# ─────────────────────────────────────────────────────────────────────────────

def get_risk_band(fake_prob: float) -> Dict[str, str]:
    """
    Assign a risk band based on fake probability.

    Args:
        fake_prob: P(fake) ∈ [0, 1].

    Returns:
        Dict with 'level', 'emoji', 'color', 'description'.
    """
    if fake_prob <= RISK_LOW_MAX:
        return {
            "level":       "LOW",
            "emoji":       "🟢",
            "color":       "#2A9D8F",
            "description": "Audio appears authentic with high confidence.",
        }
    elif fake_prob <= RISK_MED_MAX:
        return {
            "level":       "MEDIUM",
            "emoji":       "🟡",
            "color":       "#E9C46A",
            "description": "Uncertain — manual review recommended.",
        }
    else:
        return {
            "level":       "HIGH",
            "emoji":       "🔴",
            "color":       "#E63946",
            "description": "Strong indicators of AI-generated/spoofed audio.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# EFFICIENTNET PREPROCESSING FOR INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_for_efficientnet(spec: np.ndarray) -> np.ndarray:
    """
    Convert (N_MELS, T_FRAMES, 1) spec → (1, 224, 224, 3) for EfficientNetB0.
    """
    from PIL import Image
    import numpy as np

    # Squeeze and normalise to [0, 255]
    spec_2d = spec.squeeze()  # (N_MELS, T_FRAMES)
    spec_norm = (spec_2d - spec_2d.min()) / (spec_2d.max() - spec_2d.min() + 1e-8)
    spec_uint8 = (spec_norm * 255).astype(np.uint8)

    # Resize to 224×224
    img = Image.fromarray(spec_uint8, mode="L").convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)  # (224, 224, 3) in [0, 255]

    return arr[np.newaxis, ...]  # (1, 224, 224, 3)


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AudioInferenceEngine:
    """
    Production inference engine for DeepShield Audio models.

    Supports:
      - Single-file inference (short clips ≤ 4 s)
      - Long-audio windowed inference (sliding window)
      - Grad-CAM explainability integration
    """

    def __init__(
        self,
        model,
        model_name: str = "custom_cnn",
        threshold: Optional[float] = None,
        sr: int = SAMPLE_RATE,
    ):
        """
        Args:
            model:       Loaded Keras model (compiled).
            model_name:  One of 'custom_cnn', 'cnn_lstm', 'efficientnet_b0'.
            threshold:   Decision threshold. If None, loads from store or uses default.
            sr:          Expected sample rate.
        """
        self.model      = model
        self.model_name = model_name
        self.sr         = sr
        self.is_efficientnet = "efficient" in model_name.lower()

        self.threshold = (
            threshold
            if threshold is not None
            else load_optimal_threshold(model_name, fallback=DEFAULT_THRESHOLD)
        )
        logger.info("InferenceEngine ready | model=%s | threshold=%.3f",
                    model_name, self.threshold)

    def _preprocess(self, spec: np.ndarray) -> np.ndarray:
        """
        Apply model-specific preprocessing to a (N_MELS, T_FRAMES, 1) spec.
        Returns a batch tensor ready for model().
        """
        if self.is_efficientnet:
            return _preprocess_for_efficientnet(spec)
        else:
            return spec[np.newaxis, ...]   # (1, N_MELS, T_FRAMES, 1)

    def _raw_score(self, spec_batch: np.ndarray) -> float:
        """
        Run forward pass and return raw P(real) score.
        """
        import tensorflow as tf
        scores = self.model(
            tf.cast(spec_batch, tf.float32), training=False
        ).numpy().flatten()
        return float(scores[0])

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        path: Union[str, Path],
        return_gradcam: bool = False,
    ) -> Dict:
        """
        Single-file audio deepfake detection.

        Args:
            path:           Path to audio file (WAV / FLAC / MP3).
            return_gradcam: If True, compute Grad-CAM overlay.

        Returns:
            Dict with:
              'label':       'bonafide' or 'spoof'
              'is_fake':     bool
              'real_prob':   float ∈ [0, 1]
              'fake_prob':   float ∈ [0, 1]
              'confidence':  max(real_prob, fake_prob)
              'threshold':   decision threshold used
              'risk':        risk band dict
              'spec':        raw spectrogram (H, W, 1)
              'gradcam':     gradcam result dict (if requested)
              'model_name':  str
              'inference_ms': wall-clock time in ms
        """
        t0 = time.perf_counter()

        spec      = preprocess_file(str(path), add_channel_dim=True)
        batch     = self._preprocess(spec)
        real_prob = self._raw_score(batch)
        fake_prob = 1.0 - real_prob

        is_fake  = fake_prob >= self.threshold
        label    = "spoof" if is_fake else "bonafide"
        conf     = float(max(real_prob, fake_prob))
        risk     = get_risk_band(fake_prob)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = {
            "label":        label,
            "is_fake":      bool(is_fake),
            "real_prob":    float(real_prob),
            "fake_prob":    float(fake_prob),
            "confidence":   conf,
            "threshold":    self.threshold,
            "risk":         risk,
            "spec":         spec,
            "model_name":   self.model_name,
            "inference_ms": elapsed_ms,
        }

        if return_gradcam:
            try:
                from src.explainability import get_gradcam_for_model
                gradcam = get_gradcam_for_model(self.model, self.model_name)
                result["gradcam"] = gradcam.explain(spec)
            except Exception as e:
                logger.warning("Grad-CAM failed: %s", e)
                result["gradcam"] = None

        logger.info(
            "predict() | %s | %.1f ms | P(fake)=%.4f | label=%s | risk=%s",
            Path(path).name, elapsed_ms, fake_prob, label, risk["level"],
        )
        return result

    def predict_long_audio(
        self,
        path: Union[str, Path],
        window_sec: float = WINDOW_SECONDS,
        hop_sec: float = HOP_SECONDS,
    ) -> Dict:
        """
        Windowed inference for long audio (> 4 seconds).

        Slices the audio into overlapping windows, runs predict() on each,
        and aggregates results.

        Args:
            path:       Path to audio file.
            window_sec: Window duration in seconds.
            hop_sec:    Step size between windows (< window_sec for overlap).

        Returns:
            Dict with:
              'windows':       list of per-window result dicts
              'aggregate_fake_prob': mean P(fake) across all windows
              'overall_label':      final aggregated label
              'overall_is_fake':    bool
              'overall_confidence': float
              'risk':               overall risk band dict
              'duration_sec':       audio duration in seconds
              'n_windows':          int
        """
        t0 = time.perf_counter()

        waveform = load_audio(str(path), sr=self.sr)
        duration_sec = len(waveform) / self.sr

        window_samples = int(window_sec * self.sr)
        hop_samples    = int(hop_sec   * self.sr)

        window_results = []
        start = 0
        win_idx = 0

        while start < len(waveform):
            end    = min(start + window_samples, len(waveform))
            chunk  = waveform[start:end]

            spec   = preprocess_waveform(chunk, sr=self.sr, add_channel_dim=True)
            batch  = self._preprocess(spec)
            real_p = self._raw_score(batch)
            fake_p = 1.0 - real_p

            window_results.append({
                "window_idx": win_idx,
                "start_sec":  round(start / self.sr, 3),
                "end_sec":    round(end   / self.sr, 3),
                "real_prob":  float(real_p),
                "fake_prob":  float(fake_p),
                "label":      "spoof" if fake_p >= self.threshold else "bonafide",
                "is_fake":    bool(fake_p >= self.threshold),
            })

            if end == len(waveform):
                break
            start   += hop_samples
            win_idx += 1

        # Aggregation: mean fake probability
        fake_probs    = [w["fake_prob"] for w in window_results]
        agg_fake_prob = float(np.mean(fake_probs))
        agg_real_prob = 1.0 - agg_fake_prob
        overall_fake  = agg_fake_prob >= self.threshold
        conf          = float(max(agg_fake_prob, agg_real_prob))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "predict_long() | %s | %.1f s | %d windows | %.1f ms | P(fake)=%.4f",
            Path(path).name, duration_sec, len(window_results), elapsed_ms, agg_fake_prob,
        )

        return {
            "windows":              window_results,
            "aggregate_fake_prob":  agg_fake_prob,
            "aggregate_real_prob":  agg_real_prob,
            "overall_label":        "spoof" if overall_fake else "bonafide",
            "overall_is_fake":      bool(overall_fake),
            "overall_confidence":   conf,
            "risk":                 get_risk_band(agg_fake_prob),
            "duration_sec":         duration_sec,
            "n_windows":            len(window_results),
            "model_name":           self.model_name,
            "inference_ms":         elapsed_ms,
        }

    def predict_from_waveform(
        self,
        waveform: np.ndarray,
        sr: int = SAMPLE_RATE,
    ) -> Dict:
        """
        Run inference directly from a numpy waveform (e.g., from Streamlit upload).

        Args:
            waveform: 1-D float32 audio samples.
            sr:       Sample rate of waveform.

        Returns:
            Same format as predict().
        """
        import librosa
        t0 = time.perf_counter()

        # Resample if needed
        if sr != self.sr:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sr)

        spec      = preprocess_waveform(waveform, sr=self.sr, add_channel_dim=True)
        batch     = self._preprocess(spec)
        real_prob = self._raw_score(batch)
        fake_prob = 1.0 - real_prob

        is_fake   = fake_prob >= self.threshold
        label     = "spoof" if is_fake else "bonafide"
        conf      = float(max(real_prob, fake_prob))

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "label":        label,
            "is_fake":      bool(is_fake),
            "real_prob":    float(real_prob),
            "fake_prob":    float(fake_prob),
            "confidence":   conf,
            "threshold":    self.threshold,
            "risk":         get_risk_band(fake_prob),
            "spec":         spec,
            "model_name":   self.model_name,
            "inference_ms": elapsed_ms,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS (non-OOP interface)
# ─────────────────────────────────────────────────────────────────────────────

def predict_audio(
    path: Union[str, Path],
    model,
    model_name: str = "custom_cnn",
    threshold: Optional[float] = None,
) -> Dict:
    """Functional wrapper around AudioInferenceEngine.predict()."""
    engine = AudioInferenceEngine(model, model_name=model_name, threshold=threshold)
    return engine.predict(path)


def predict_long_audio(
    path: Union[str, Path],
    model,
    model_name: str = "custom_cnn",
    threshold: Optional[float] = None,
    window_sec: float = WINDOW_SECONDS,
    hop_sec: float = HOP_SECONDS,
) -> Dict:
    """Functional wrapper around AudioInferenceEngine.predict_long_audio()."""
    engine = AudioInferenceEngine(model, model_name=model_name, threshold=threshold)
    return engine.predict_long_audio(path, window_sec=window_sec, hop_sec=hop_sec)


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="DeepShield Audio — Inference CLI")
    parser.add_argument("audio",        type=str, help="Path to audio file")
    parser.add_argument("--model",      type=str, default="custom_cnn",
                        choices=["custom_cnn", "cnn_lstm", "efficientnet_b0"])
    parser.add_argument("--threshold",  type=float, default=None)
    parser.add_argument("--windowed",   action="store_true",
                        help="Use windowed inference for long audio")
    parser.add_argument("--window-sec", type=float, default=WINDOW_SECONDS)
    args = parser.parse_args()

    from src.utils import setup_logging
    setup_logging("INFO")

    from src.trainer import load_model
    model  = load_model(args.model)
    engine = AudioInferenceEngine(model, model_name=args.model,
                                  threshold=args.threshold)

    if args.windowed:
        result = engine.predict_long_audio(args.audio, window_sec=args.window_sec)
        # Don't print spec arrays
        printable = {k: v for k, v in result.items() if k != "windows"}
        print(json.dumps(printable, indent=2))
        print(f"\nWindow breakdown ({result['n_windows']} windows):")
        for w in result["windows"]:
            bar = "█" * int(w["fake_prob"] * 20)
            print(f"  [{w['start_sec']:.1f}s–{w['end_sec']:.1f}s] "
                  f"P(fake)={w['fake_prob']:.3f} {bar} {w['label']}")
    else:
        result = engine.predict(args.audio)
        printable = {k: v for k, v in result.items()
                     if k not in ("spec", "gradcam")}
        print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
