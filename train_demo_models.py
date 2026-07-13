"""
Quick script to train all 3 DeepShield models on synthetic dummy data
and save them to the correct paths so the Streamlit app can load them.

Run with: .venv/bin/python train_demo_models.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "tensorflow"

import sys
sys.path.insert(0, ".")

import numpy as np
from pathlib import Path

from src.config import N_MELS, T_FRAMES, SAVED_MODELS_DIR, LOGS_DIR, EFFICIENTNET_INPUT_SIZE
from src.models.custom_cnn import build_custom_cnn
from src.models.cnn_lstm import build_cnn_lstm
from src.models.efficientnet import build_efficientnet_b0
from src.trainer import build_callbacks

print("=" * 60)
print("  DeepShield Audio — Demo Model Training")
print("  (Training on synthetic data so the app loads properly)")
print("=" * 60)

EPOCHS = 3
BATCH  = 16

# ── Synthetic data for CNN / LSTM ─────────────────────────────────────────────
N = 128
X_cnn  = np.random.randn(N, N_MELS, T_FRAMES, 1).astype("float32")
y      = np.random.randint(0, 2, size=(N, 1)).astype("float32")

# ── 1. Custom CNN ─────────────────────────────────────────────────────────────
print("\n[1/3] Training Custom CNN …")
model_dir = SAVED_MODELS_DIR / "custom_cnn"
model_dir.mkdir(parents=True, exist_ok=True)

cnn = build_custom_cnn()
cb = build_callbacks(
    model_name="custom_cnn",
    checkpoint_dir=model_dir,
    tensorboard_dir=LOGS_DIR,
)
cnn.fit(X_cnn, y, batch_size=BATCH, epochs=EPOCHS,
        validation_split=0.2, callbacks=cb, verbose=1)
print(f"  ✅ Saved → {model_dir / 'custom_cnn_best.keras'}")

# ── 2. CNN-LSTM ───────────────────────────────────────────────────────────────
print("\n[2/3] Training CNN-LSTM …")
model_dir = SAVED_MODELS_DIR / "cnn_lstm"
model_dir.mkdir(parents=True, exist_ok=True)

lstm = build_cnn_lstm()
cb = build_callbacks(
    model_name="cnn_lstm",
    checkpoint_dir=model_dir,
    tensorboard_dir=LOGS_DIR,
)
lstm.fit(X_cnn, y, batch_size=BATCH, epochs=EPOCHS,
         validation_split=0.2, callbacks=cb, verbose=1)
print(f"  ✅ Saved → {model_dir / 'cnn_lstm_best.keras'}")

# ── 3. EfficientNetB0 ─────────────────────────────────────────────────────────
print("\n[3/3] Training EfficientNetB0 …")
model_dir = SAVED_MODELS_DIR / "efficientnet_b0"
model_dir.mkdir(parents=True, exist_ok=True)

# EfficientNet needs (224, 224, 3) input
X_eff = np.random.randn(N, 224, 224, 3).astype("float32")
eff = build_efficientnet_b0(freeze_backbone=True)
cb = build_callbacks(
    model_name="efficientnet_b0",
    checkpoint_dir=model_dir,
    tensorboard_dir=LOGS_DIR,
)
eff.fit(X_eff, y, batch_size=BATCH, epochs=EPOCHS,
        validation_split=0.2, callbacks=cb, verbose=1)
print(f"  ✅ Saved → {model_dir / 'efficientnet_b0_best.keras'}")

print("\n" + "=" * 60)
print("  All 3 models saved! Reload the Streamlit app now.")
print("=" * 60)
