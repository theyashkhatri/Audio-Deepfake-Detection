"""
DeepShield Audio — CNN-LSTM Hybrid Model
==========================================
Combines convolutional feature extraction with sequential modelling via
a Bidirectional LSTM layer to capture temporal dependencies in speech.

Architecture:
  Input: (128, T, 1)

  CNN Feature Extractor:
    Conv2D(64,  3×3) → BN → ReLU → MaxPool(2×2)
    Conv2D(128, 3×3) → BN → ReLU → MaxPool(2×2)
    Conv2D(128, 3×3) → BN → ReLU                 ← Grad-CAM target

  Reshape to sequence:
    (batch, time_steps, features)  — flatten freq × channels per timestep

  Sequential Modelling:
    Bidirectional LSTM(128, return_sequences=True)
    LSTM(64, return_sequences=False)
    Dropout(0.4)

  Head:
    Dense(128, ReLU) → Dropout(0.3) → Dense(1, sigmoid)

Rationale: CNNs extract local spectro-temporal patterns; LSTM models
long-range temporal structure (prosody, rhythm) crucial for detecting TTS/VC.
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


def build_cnn_lstm(
    input_shape: tuple = (N_MELS, T_FRAMES, 1),
    learning_rate: float = LEARNING_RATE,
    lstm_units_1: int = 128,
    lstm_units_2: int = 64,
    l2_reg: float = 1e-4,
) -> keras.Model:
    """
    Build and compile the CNN-LSTM hybrid model.

    Args:
        input_shape:   (N_MELS, T_FRAMES, 1)
        learning_rate: Adam learning rate.
        lstm_units_1:  Bidirectional LSTM units (halved per direction).
        lstm_units_2:  Second LSTM units.
        l2_reg:        L2 weight decay.

    Returns:
        Compiled Keras Model.
    """
    reg = keras.regularizers.l2(l2_reg)

    inputs = keras.Input(shape=input_shape, name="log_mel_input")

    # ── CNN Feature Extraction ───────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg,
                      name="cnn_conv1")(inputs)
    x = layers.BatchNormalization(name="cnn_bn1")(x)
    x = layers.Activation("relu", name="cnn_relu1")(x)
    x = layers.MaxPooling2D((2, 4), name="cnn_pool1")(x)   # (64, ~63, 32)
    x = layers.Dropout(0.2, name="cnn_drop1")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg,
                      name="cnn_conv2")(x)
    x = layers.BatchNormalization(name="cnn_bn2")(x)
    x = layers.Activation("relu", name="cnn_relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="cnn_pool2")(x)   # (32, ~31, 64)
    x = layers.Dropout(0.2, name="cnn_drop2")(x)

    # Grad-CAM target layer
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg,
                      name="cnn_conv3_gradcam")(x)
    x = layers.BatchNormalization(name="cnn_bn3")(x)
    x = layers.Activation("relu", name="cnn_relu3")(x)
    x = layers.Dropout(0.25, name="cnn_drop3")(x)

    # ── Reshape for LSTM: merge freq and channel dims ─────────────────────────
    # x shape: (batch, freq_bins, time_steps, channels)
    # Permute → (batch, time_steps, freq_bins, channels)
    x = layers.Permute((2, 1, 3), name="permute")(x)
    # Merge freq × channels into a flat feature vector per timestep
    shape = x.shape
    x = layers.Reshape((shape[1], shape[2] * shape[3]), name="reshape_seq")(x)
    # x shape: (batch, time_steps, features)

    # ── Sequential Modelling ─────────────────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(lstm_units_1 // 2, return_sequences=True,
                    kernel_regularizer=reg, name="lstm_fwd"),
        name="bi_lstm1",
    )(x)
    x = layers.Dropout(0.3, name="lstm_drop1")(x)

    x = layers.LSTM(lstm_units_2, return_sequences=False,
                    kernel_regularizer=reg, name="lstm2")(x)
    x = layers.Dropout(0.4, name="lstm_drop2")(x)

    # ── Head ─────────────────────────────────────────────────────────────────
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg,
                     name="fc1")(x)
    x = layers.Dropout(0.3, name="fc_drop")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="DeepShield_CNN_LSTM")

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
    model = build_cnn_lstm()
    model.summary()
    print(f"\nTotal trainable params: {model.count_params():,}")
