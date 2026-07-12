"""
DeepShield Audio — Audio Preprocessor
=======================================
SINGLE SOURCE OF TRUTH for audio → Log-Mel spectrogram conversion.
This exact module is used at BOTH train-time AND inference-time.
Never duplicate this logic — always import from here.

Pipeline:
  1. Load audio at 16 kHz (resample if needed)
  2. Pad or trim to MAX_AUDIO_SAMPLES
  3. Extract Log-Mel spectrogram  →  shape (N_MELS, T_FRAMES)
  4. Normalise to [−1, 1] using train-time global statistics
  5. Add channel dim  →  shape (N_MELS, T_FRAMES, 1)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import librosa

try:
    from src.config import (
        SAMPLE_RATE, MAX_AUDIO_SAMPLES, N_MELS,
        N_FFT, HOP_LENGTH, WIN_LENGTH, FMIN, FMAX,
        T_FRAMES, SPEC_SHAPE,
    )
except ImportError:
    from config import (
        SAMPLE_RATE, MAX_AUDIO_SAMPLES, N_MELS,
        N_FFT, HOP_LENGTH, WIN_LENGTH, FMIN, FMAX,
        T_FRAMES, SPEC_SHAPE,
    )

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# NORMALISATION STATISTICS (computed on train split — updated after EDA)
# These defaults are reasonable starting points; run compute_global_stats()
# on your training data and persist the result to override them.
# ─────────────────────────────────────────────────────────────────────────────
_GLOBAL_MEAN: float = -40.0   # dB — approximate mean over ASVspoof2019 train
_GLOBAL_STD:  float =  20.0   # dB — approximate std  over ASVspoof2019 train


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(
    path: Union[str, Path],
    sr: int = SAMPLE_RATE,
    mono: bool = True,
) -> np.ndarray:
    """
    Load audio from file (WAV, FLAC, MP3) and resample to `sr`.

    Args:
        path: Path to audio file.
        sr:   Target sample rate (default 16 000 Hz).
        mono: Mix to mono if True.

    Returns:
        1-D float32 waveform array, shape (n_samples,).
    """
    path = str(path)
    try:
        waveform, _sr = librosa.load(path, sr=sr, mono=mono, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio file: {path}") from exc
    return waveform


def pad_or_trim(waveform: np.ndarray, target_length: int = MAX_AUDIO_SAMPLES) -> np.ndarray:
    """
    Pad (with zeros) or trim a waveform to exactly `target_length` samples.

    Args:
        waveform:      1-D float32 array.
        target_length: Desired number of samples.

    Returns:
        1-D float32 array of length exactly `target_length`.
    """
    n = len(waveform)
    if n >= target_length:
        return waveform[:target_length]
    # Zero-pad on the right
    padded = np.zeros(target_length, dtype=np.float32)
    padded[:n] = waveform
    return padded


def extract_log_mel(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """
    Compute Log-Mel spectrogram from a waveform.

    Args:
        waveform: 1-D float32 audio samples.
        sr: Sample rate.
        n_mels, n_fft, hop_length, win_length, fmin, fmax: STFT/mel params.

    Returns:
        2-D float32 array of shape (n_mels, T).
        Values are in dB scale (log-mel energies).
    """
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    return log_mel.astype(np.float32)


def pad_or_trim_spectrogram(
    spec: np.ndarray,
    target_frames: int = T_FRAMES,
) -> np.ndarray:
    """
    Pad (right side) or trim (right side) a spectrogram to `target_frames` columns.

    Args:
        spec:          2-D array of shape (n_mels, T).
        target_frames: Target number of time frames.

    Returns:
        2-D array of shape (n_mels, target_frames).
    """
    n_mels, t = spec.shape
    if t >= target_frames:
        return spec[:, :target_frames]
    pad_width = target_frames - t
    return np.pad(spec, ((0, 0), (0, pad_width)), mode="constant", constant_values=spec.min())


def normalise_spectrogram(
    spec: np.ndarray,
    mean: float = _GLOBAL_MEAN,
    std: float = _GLOBAL_STD,
) -> np.ndarray:
    """
    Standardise spectrogram using global mean/std computed from train split.

    Args:
        spec: 2-D spectrogram (n_mels, T), dB scale.
        mean: Global mean dB value (computed on train split ONLY).
        std:  Global std dB value  (computed on train split ONLY).

    Returns:
        Normalised float32 array, same shape.
    """
    return ((spec - mean) / (std + 1e-8)).astype(np.float32)


def set_global_stats(mean: float, std: float) -> None:
    """
    Update module-level global mean/std (called once after computing from train set).
    """
    global _GLOBAL_MEAN, _GLOBAL_STD
    _GLOBAL_MEAN = mean
    _GLOBAL_STD  = std
    logger.info("Global stats updated: mean=%.3f, std=%.3f", mean, std)


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL PIPELINE (used by both training & inference)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_file(
    path: Union[str, Path],
    sr: int = SAMPLE_RATE,
    target_samples: int = MAX_AUDIO_SAMPLES,
    target_frames: int = T_FRAMES,
    normalise: bool = True,
    add_channel_dim: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline: file → model-ready spectrogram tensor.

    Args:
        path:           Path to audio file (WAV / FLAC / MP3).
        sr:             Target sample rate.
        target_samples: Pad/trim length in samples.
        target_frames:  Pad/trim spectrogram width.
        normalise:      Apply global standardisation.
        add_channel_dim: Append trailing channel dim (for CNN input).

    Returns:
        numpy array:
          - Without channel: shape (N_MELS, T_FRAMES)
          - With channel:    shape (N_MELS, T_FRAMES, 1)
    """
    waveform = load_audio(path, sr=sr)
    waveform = pad_or_trim(waveform, target_length=target_samples)
    spec     = extract_log_mel(waveform, sr=sr)
    spec     = pad_or_trim_spectrogram(spec, target_frames=target_frames)

    if normalise:
        spec = normalise_spectrogram(spec)

    if add_channel_dim:
        spec = spec[..., np.newaxis]   # (N_MELS, T_FRAMES, 1)

    return spec


def preprocess_waveform(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    target_samples: int = MAX_AUDIO_SAMPLES,
    target_frames: int = T_FRAMES,
    normalise: bool = True,
    add_channel_dim: bool = True,
) -> np.ndarray:
    """
    Preprocessing pipeline starting from raw waveform (already loaded).
    Useful for windowed inference where we slice the waveform first.

    Returns same shape as preprocess_file().
    """
    waveform = pad_or_trim(waveform, target_length=target_samples)
    spec     = extract_log_mel(waveform, sr=sr)
    spec     = pad_or_trim_spectrogram(spec, target_frames=target_frames)

    if normalise:
        spec = normalise_spectrogram(spec)

    if add_channel_dim:
        spec = spec[..., np.newaxis]

    return spec


# ─────────────────────────────────────────────────────────────────────────────
# STAT COMPUTATION (run once on train split)
# ─────────────────────────────────────────────────────────────────────────────

def compute_global_stats(
    file_paths: list,
    sr: int = SAMPLE_RATE,
    target_samples: int = MAX_AUDIO_SAMPLES,
    target_frames: int = T_FRAMES,
    max_files: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute mean and std of Log-Mel spectrograms across a list of files.
    Should be called ONLY on training files to prevent data leakage.

    Args:
        file_paths: List of audio file paths.
        max_files:  Subsample if dataset is large (e.g. 5000 for fast estimate).

    Returns:
        (mean, std) as floats.
    """
    from tqdm import tqdm

    if max_files and len(file_paths) > max_files:
        rng = np.random.default_rng(42)
        file_paths = list(rng.choice(file_paths, size=max_files, replace=False))

    all_values = []
    for path in tqdm(file_paths, desc="Computing global stats", unit="file"):
        try:
            wav  = load_audio(path, sr=sr)
            wav  = pad_or_trim(wav, target_length=target_samples)
            spec = extract_log_mel(wav, sr=sr)
            spec = pad_or_trim_spectrogram(spec, target_frames=target_frames)
            all_values.append(spec.ravel())
        except Exception as e:
            logger.warning("Skipping %s: %s", path, e)

    if not all_values:
        logger.warning("No files processed — returning defaults.")
        return _GLOBAL_MEAN, _GLOBAL_STD

    concatenated = np.concatenate(all_values)
    mean = float(np.mean(concatenated))
    std  = float(np.std(concatenated))
    logger.info("Global stats — mean: %.4f, std: %.4f", mean, std)
    return mean, std
