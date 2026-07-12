"""
DeepShield Audio — EfficientNetB0 Transfer Learning Model
===========================================================
Adapts the ImageNet-pretrained EfficientNetB0 backbone for audio
deepfake detection by treating Log-Mel spectrograms as images.

Architecture:
  Input: (224, 224, 3)  — resized + 3-channel spectrogram (rescaled to [0,255])

  Phase 1 — Feature Extraction (backbone frozen):
    EfficientNetB0 (pretrained, all layers frozen)
    → GlobalAveragePooling2D
    → Dense(256, ReLU) + Dropout(0.4)
    → Dense(1, sigmoid)
    Train for a few epochs (let head learn)

  Phase 2 — Fine-tuning (top EfficientNet blocks unfrozen):
    Unfreeze from 'block5a' onwards (~40% of backbone)
    → Lower learning rate (1e-5)
    → Continue training

Rationale: EfficientNetB0's compound scaling makes it an ideal
backbone for 2D feature extraction even on spectrograms. ImageNet
edge/texture detectors transfer surprisingly well to mel-spectrograms.
"""

import tensorflow as tf
try:
    # Keras 3 (TF 2.16+)
    import keras
    from keras import layers
except ImportError:
    # Keras 2 (TF 2.15)
    from tensorflow import keras
    from tensorflow.keras import layers

try:
    from src.config import EFFICIENTNET_INPUT_SIZE, LEARNING_RATE
except ImportError:
    from config import EFFICIENTNET_INPUT_SIZE, LEARNING_RATE

# Layer at which fine-tuning begins (all layers from here onwards are unfrozen)
FINETUNE_FROM_LAYER = "block5a_expand_conv"


def build_efficientnet_b0(
    input_shape: tuple = EFFICIENTNET_INPUT_SIZE,  # (128, 251, 3) — raw
    learning_rate: float = LEARNING_RATE,
    dropout_rate: float = 0.4,
    freeze_backbone: bool = True,
    l2_reg: float = 1e-4,
) -> keras.Model:
    """
    Build EfficientNetB0 transfer learning model.

    Args:
        input_shape:      (H, W, 3) input — typically (224, 224, 3) after resize.
        learning_rate:    Adam learning rate for Phase 1.
        dropout_rate:     Dropout rate in the head.
        freeze_backbone:  If True, backbone is frozen (Phase 1).
                          Set False for Phase 2 fine-tuning.
        l2_reg:           L2 regularisation on Dense layers.

    Returns:
        Compiled Keras Model.
    """
    reg = keras.regularizers.l2(l2_reg)

    # EfficientNetB0 expects 224×224 — we use the dataset pipeline to resize
    # Here input_shape may be the pre-resize shape; actual backbone always gets 224×224
    effective_input_shape = (224, 224, 3)

    inputs = keras.Input(shape=effective_input_shape, name="spectrogram_rgb_input")

    # ── EfficientNetB0 Backbone ───────────────────────────────────────────────
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        input_shape=effective_input_shape,
    )
    backbone.trainable = not freeze_backbone

    # ── Head ─────────────────────────────────────────────────────────────────
    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg,
                     name="fc1")(x)
    x = layers.Dropout(dropout_rate, name="drop1")(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=reg,
                     name="fc2")(x)
    x = layers.Dropout(dropout_rate / 2, name="drop2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="DeepShield_EfficientNetB0")

    _compile(model, learning_rate)
    return model


def unfreeze_for_finetuning(
    model: keras.Model,
    finetune_from: str = FINETUNE_FROM_LAYER,
    learning_rate: float = 1e-5,
) -> keras.Model:
    """
    Phase 2: Unfreeze the top portion of the EfficientNetB0 backbone
    for fine-tuning. Should be called after Phase 1 training converges.

    Args:
        model:          The already-trained Phase 1 model.
        finetune_from:  Layer name from which to unfreeze (inclusive).
        learning_rate:  Reduced learning rate for fine-tuning.

    Returns:
        Re-compiled model with partial backbone unfrozen.
    """
    # Find backbone (EfficientNetB0 is a sub-model layer)
    backbone = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            backbone = layer
            break

    if backbone is None:
        # Fallback: unfreeze all layers beyond head
        model.trainable = True
        _compile(model, learning_rate)
        return model

    backbone.trainable = True

    # Freeze everything before the target layer
    set_trainable = False
    for layer in backbone.layers:
        if layer.name == finetune_from:
            set_trainable = True
        layer.trainable = set_trainable

    _compile(model, learning_rate)

    n_trainable = sum(1 for l in backbone.layers if l.trainable)
    n_total     = len(backbone.layers)
    print(f"Fine-tuning: {n_trainable}/{n_total} backbone layers unfrozen.")
    return model


def _compile(model: keras.Model, learning_rate: float) -> None:
    """Compile with Adam, binary cross-entropy, and standard metrics."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


if __name__ == "__main__":
    model = build_efficientnet_b0(freeze_backbone=True)
    model.summary()
    print(f"\nTotal trainable params: {model.count_params():,}")
    n_trainable = sum(1 for l in model.layers if l.trainable)
    print(f"Trainable layers: {n_trainable}/{len(model.layers)}")
