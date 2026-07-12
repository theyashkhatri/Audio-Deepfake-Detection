"""
Tests for src/models/ — build and forward pass verification.

These tests ensure:
  - Models build without error
  - Output shapes are correct
  - Models are differentiable (gradients flow)
  - Summary string is non-empty
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dummy_cnn_batch():
    """Random batch for CNN / CNN-LSTM models: (B, N_MELS, T_FRAMES, 1)."""
    from src.config import N_MELS, T_FRAMES
    rng = np.random.default_rng(0)
    return rng.standard_normal((2, N_MELS, T_FRAMES, 1)).astype(np.float32)


@pytest.fixture(scope="module")
def dummy_efficientnet_batch():
    """Random batch for EfficientNetB0: (B, 224, 224, 3)."""
    rng = np.random.default_rng(0)
    return (rng.random((2, 224, 224, 3)) * 255.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CNN
# ─────────────────────────────────────────────────────────────────────────────

class TestCustomCNN:
    @pytest.fixture(scope="class")
    def model(self):
        from src.models.custom_cnn import build_custom_cnn
        return build_custom_cnn()

    def test_model_builds(self, model):
        assert model is not None

    def test_output_shape(self, model, dummy_cnn_batch):
        import tensorflow as tf
        out = model(dummy_cnn_batch, training=False)
        assert out.shape == (2, 1), f"Expected (2,1), got {out.shape}"

    def test_output_in_01_range(self, model, dummy_cnn_batch):
        import tensorflow as tf
        out = model(dummy_cnn_batch, training=False).numpy()
        assert np.all(out >= 0.0) and np.all(out <= 1.0), \
            "Sigmoid output must be in [0, 1]"

    def test_has_trainable_params(self, model):
        assert model.count_params() > 0

    def test_gradcam_layer_exists(self, model):
        layer_names = [l.name for l in model.layers]
        assert "conv4_gradcam" in layer_names, \
            "Grad-CAM target layer 'conv4_gradcam' not found"

    def test_gradient_flows(self, model, dummy_cnn_batch):
        import tensorflow as tf
        inp = tf.Variable(dummy_cnn_batch)
        with tf.GradientTape() as tape:
            out  = model(inp, training=True)
            loss = tf.reduce_mean(out)
        grads = tape.gradient(loss, model.trainable_variables)
        non_none = [g for g in grads if g is not None]
        assert len(non_none) > 0, "No gradients computed"

    def test_model_summary_nonempty(self, model, capsys):
        model.summary()
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_batch_sizes(self, model):
        from src.config import N_MELS, T_FRAMES
        rng = np.random.default_rng(1)
        for bs in [1, 4, 8]:
            x   = rng.standard_normal((bs, N_MELS, T_FRAMES, 1)).astype(np.float32)
            out = model(x, training=False)
            assert out.shape == (bs, 1)


# ─────────────────────────────────────────────────────────────────────────────
# CNN-LSTM
# ─────────────────────────────────────────────────────────────────────────────

class TestCNNLSTM:
    @pytest.fixture(scope="class")
    def model(self):
        from src.models.cnn_lstm import build_cnn_lstm
        return build_cnn_lstm()

    def test_model_builds(self, model):
        assert model is not None

    def test_output_shape(self, model, dummy_cnn_batch):
        out = model(dummy_cnn_batch, training=False)
        assert out.shape == (2, 1)

    def test_output_sigmoid(self, model, dummy_cnn_batch):
        out = model(dummy_cnn_batch, training=False).numpy()
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_gradcam_layer_exists(self, model):
        layer_names = [l.name for l in model.layers]
        assert "cnn_conv3_gradcam" in layer_names, \
            "Grad-CAM target layer 'cnn_conv3_gradcam' not found"

    def test_has_lstm_layers(self, model):
        import tensorflow as tf
        lstm_layers = [l for l in model.layers
                       if isinstance(l, (tf.keras.layers.LSTM,
                                        tf.keras.layers.Bidirectional))]
        assert len(lstm_layers) >= 1, "Expected at least one LSTM/Bidirectional layer"

    def test_trainable_params(self, model):
        assert model.count_params() > 0

    def test_gradient_flows(self, model, dummy_cnn_batch):
        import tensorflow as tf
        inp = tf.Variable(dummy_cnn_batch)
        with tf.GradientTape() as tape:
            out  = model(inp, training=True)
            loss = tf.reduce_mean(out)
        grads    = tape.gradient(loss, model.trainable_variables)
        non_none = [g for g in grads if g is not None]
        assert len(non_none) > 0


# ─────────────────────────────────────────────────────────────────────────────
# EFFICIENTNET B0
# ─────────────────────────────────────────────────────────────────────────────

class TestEfficientNetB0:
    @pytest.fixture(scope="class")
    def model(self):
        from src.models.efficientnet import build_efficientnet_b0
        # Use freeze_backbone=True to avoid downloading full weights in CI
        return build_efficientnet_b0(freeze_backbone=True)

    def test_model_builds(self, model):
        assert model is not None

    def test_output_shape(self, model, dummy_efficientnet_batch):
        out = model(dummy_efficientnet_batch, training=False)
        assert out.shape == (2, 1)

    def test_output_sigmoid(self, model, dummy_efficientnet_batch):
        out = model(dummy_efficientnet_batch, training=False).numpy()
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_backbone_frozen(self, model):
        """When freeze_backbone=True, backbone layers should not be trainable."""
        backbone = None
        for layer in model.layers:
            if "efficientnet" in layer.name.lower():
                backbone = layer
                break
        if backbone is not None:
            # At least some backbone layers should be non-trainable
            non_trainable = [l for l in backbone.layers if not l.trainable]
            assert len(non_trainable) > 0, "Expected backbone to be (partially) frozen"

    def test_head_is_trainable(self, model):
        """Dense head layers must be trainable."""
        fc_layer = model.get_layer("fc1")
        assert fc_layer.trainable

    def test_total_params_positive(self, model):
        assert model.count_params() > 0

    def test_unfreeze_helper(self, model):
        from src.models.efficientnet import unfreeze_for_finetuning
        unfrozen_model = unfreeze_for_finetuning(model, learning_rate=1e-5)
        assert unfrozen_model is not None
        # After unfreezing, trainable params should increase
        # (at least some backbone layers are now trainable)
