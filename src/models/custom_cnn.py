"""
DeepShield Audio — Custom CNN Model
=====================================
A 4-block convolutional network designed specifically for Log-Mel spectrograms.

Architecture:
  Input: (128, T, 1)  — Log-Mel spectrogram with single channel

  Block 1: Conv2D(32, 3×3) → BN → ReLU → MaxPool(2×2) → Dropout(0.2)
  Block 2: Conv2D(64, 3×3) → BN → ReLU → MaxPool(2×2) → Dropout(0.2)
  Block 3: Conv2D(128,3×3) → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
  Block 4: Conv2D(256, 3×3)→ BN → ReLU → GlobalAveragePooling
  Head:    Dense(256, ReLU) → Dropout(0.5) → Dense(1, sigmoid)

Inductive bias: 2D convolutions capture both spectral (freq) and
temporal patterns simultaneously, which is ideal for spectrograms.
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
    from src.config import N_MELS, T_FRAMES, LEARNING_RATE
except ImportError:
    from config import N_MELS, T_FRAMES, LEARNING_RATE


def build_custom_cnn(
    input_shape: tuple = (N_MELS, T_FRAMES, 1),
    learning_rate: float = LEARNING_RATE,
    l2_reg: float = 1e-4,
) -> keras.Model:
    """
    Build and compile the Custom CNN model.

    Args:
        input_shape:   Shape of each input sample (height, width, channels).
        learning_rate: Initial learning rate for Adam optimiser.
        l2_reg:        L2 weight decay applied to Conv and Dense layers.

    Returns:
        Compiled Keras Model.
    """
    reg = keras.regularizers.l2(l2_reg)

    inputs = keras.Input(shape=input_shape, name="log_mel_input")

    # ── Block 1 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg,
                      name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg,
                      name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.2, name="drop1")(x)

    # ── Block 2 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg,
                      name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg,
                      name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.2, name="drop2")(x)

    # ── Block 3 ──────────────────────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg,
                      name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg,
                      name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="relu3_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(0.25, name="drop3")(x)

    # ── Block 4 (final conv — used for Grad-CAM) ─────────────────────────────
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=reg,
                      name="conv4_gradcam")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)

    # ── Head ─────────────────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg,
                     name="fc1")(x)
    x = layers.Dropout(0.5, name="drop_fc")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="DeepShield_CustomCNN")

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

    return model


if __name__ == "__main__":
    model = build_custom_cnn()
    model.summary()
    print(f"\nTotal trainable params: {model.count_params():,}")
