"""
DeepShield Audio — Central Configuration
=========================================
All hyperparameters, paths, and model settings in one place.
Import this module wherever you need configuration constants.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "ASVspoof2019_LA"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# ASVspoof 2019 LA subdirectories
TRAIN_DIR = DATASET_DIR / "ASVspoof2019_LA_train" / "flac"
DEV_DIR   = DATASET_DIR / "ASVspoof2019_LA_dev"   / "flac"
EVAL_DIR  = DATASET_DIR / "ASVspoof2019_LA_eval"  / "flac"

# Official protocol files
PROTOCOL_DIR = DATASET_DIR / "ASVspoof2019_LA_cm_protocols"
TRAIN_PROTOCOL = PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt"
DEV_PROTOCOL   = PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt"
EVAL_PROTOCOL  = PROTOCOL_DIR / "ASVspoof2019.LA.cm.eval.trl.txt"

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE       = 16000       # Hz — ASVspoof2019 native rate
MAX_AUDIO_SECONDS = 4.0         # Clips longer than this are trimmed
MAX_AUDIO_SAMPLES = int(SAMPLE_RATE * MAX_AUDIO_SECONDS)  # 64 000 samples

# Log-Mel spectrogram parameters
N_MELS      = 128              # Number of mel filterbanks
N_FFT       = 1024             # FFT window size
HOP_LENGTH  = 256              # Samples between successive frames
WIN_LENGTH  = 1024             # Window length (same as n_fft)
FMIN        = 0.0              # Min mel frequency (Hz)
FMAX        = 8000.0           # Max mel frequency (Hz) — Nyquist for 16kHz

# Derived spectrogram shape (after padding/trimming to MAX_AUDIO_SAMPLES)
# T_FRAMES = ceil(MAX_AUDIO_SAMPLES / HOP_LENGTH) + 1
import math
T_FRAMES    = math.ceil(MAX_AUDIO_SAMPLES / HOP_LENGTH) + 1   # ~251 frames
SPEC_SHAPE  = (N_MELS, T_FRAMES)   # (128, 251) — height x width

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE       = 32
EPOCHS           = 50
LEARNING_RATE    = 1e-3
SEED             = 42
VALIDATION_SPLIT = 0.1          # Only used if no official dev split

# EfficientNetB0 input size (needs 3-channel RGB-like input)
EFFICIENTNET_INPUT_SIZE = (128, 251, 3)

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_SECONDS   = 3.0          # Window size for long-audio inference (sec)
HOP_SECONDS      = 1.5          # Hop between windows (50% overlap)
DEFAULT_THRESHOLD= 0.5          # Decision threshold (updated after optimization)

# Risk bands based on fake probability
RISK_LOW_MAX     = 0.35         # P(fake) ≤ 0.35 → LOW risk
RISK_MED_MAX     = 0.65         # P(fake) ≤ 0.65 → MEDIUM risk
# P(fake) > 0.65 → HIGH risk

# ─────────────────────────────────────────────────────────────────────────────
# LABEL ENCODING
# ─────────────────────────────────────────────────────────────────────────────
LABEL_MAP = {
    "bonafide": 1,
    "spoof":    0,
}
INT_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL NAMES (used for checkpoint naming)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CUSTOM_CNN   = "custom_cnn"
MODEL_CNN_LSTM     = "cnn_lstm"
MODEL_EFFICIENTNET = "efficientnet_b0"
ALL_MODELS         = [MODEL_CUSTOM_CNN, MODEL_CNN_LSTM, MODEL_EFFICIENTNET]

# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────────────────────────────────────
GRADCAM_LAYER_MAP = {
    MODEL_CUSTOM_CNN:   "conv4_gradcam",     # Last conv in CustomCNN (Block 4)
    MODEL_CNN_LSTM:     "cnn_conv3_gradcam", # Last conv before LSTM reshape
    MODEL_EFFICIENTNET: None,               # Auto-detect last Conv2D in backbone
}

# ─────────────────────────────────────────────────────────────────────────────
# ENSURE DIRECTORIES EXIST
# ─────────────────────────────────────────────────────────────────────────────
for _dir in [SAVED_MODELS_DIR, RESULTS_DIR, LOGS_DIR, DATA_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
