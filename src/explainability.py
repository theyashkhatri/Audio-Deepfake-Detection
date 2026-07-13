"""
DeepShield Audio — Grad-CAM Explainability
============================================
Implements Gradient-weighted Class Activation Mapping (Grad-CAM)
for all three model architectures.

Grad-CAM highlights which regions of the Log-Mel spectrogram the model
focused on when making its real/fake decision — enabling human-interpretable
explanations of model behaviour.

Reference: Selvaraju et al. (2017) — "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization"
"""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
try:
    # Keras 3 (TF 2.16+)
    import keras
except ImportError:
    # Keras 2 (TF 2.15)
    from tensorflow import keras

try:
    from src.config import N_MELS, T_FRAMES, GRADCAM_LAYER_MAP
    from src.preprocessor import preprocess_file
except ImportError:
    from config import N_MELS, T_FRAMES, GRADCAM_LAYER_MAP
    from preprocessor import preprocess_file

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM CORE
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM implementation for Keras models.

    Supports any model with at least one Conv2D layer.
    """

    def __init__(
        self,
        model: keras.Model,
        layer_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Args:
            model:      Trained Keras model.
            layer_name: Name of the target Conv2D layer.
                        If None, auto-detects last Conv2D.
            model_name: Used to look up default layer from GRADCAM_LAYER_MAP.
        """
        self.model = model

        if layer_name is None and model_name in GRADCAM_LAYER_MAP:
            layer_name = GRADCAM_LAYER_MAP[model_name]

        if layer_name is None:
            layer_name = self._find_last_conv_layer()

        self.layer_name = layer_name
        self._validate_layer()

        # Build gradient model: outputs (target_conv_output, model_prediction)
        self.grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(self.layer_name).output,
                model.output,
            ],
        )
        logger.info("GradCAM initialised with layer: %s", self.layer_name)

    def _find_last_conv_layer(self) -> str:
        """Auto-detect the last Conv2D layer in the model."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                return layer.name
        raise ValueError("No Conv2D layer found in the model.")

    def _validate_layer(self) -> None:
        """Check that the target layer exists and is a Conv2D."""
        try:
            layer = self.model.get_layer(self.layer_name)
        except ValueError:
            raise ValueError(f"Layer '{self.layer_name}' not found in model. "
                             f"Available: {[l.name for l in self.model.layers]}")

    def compute_heatmap(
        self,
        spec_input: np.ndarray,
        class_idx: int = 0,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Compute raw Grad-CAM heatmap for a single spectrogram input.

        Args:
            spec_input:  Model input, shape (H, W, C) or (1, H, W, C).
            class_idx:   Output neuron index (0 for sigmoid binary).
            eps:         Small constant for numerical stability.

        Returns:
            2-D float32 heatmap, shape = conv layer's (H', W').
            Values ∈ [0, 1].
        """
        # Add batch dimension if needed
        if spec_input.ndim == 3:
            spec_input = spec_input[np.newaxis, ...]

        inp_tensor = tf.cast(spec_input, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(inp_tensor)
            conv_outputs, predictions = self.grad_model(inp_tensor, training=False)
            # For binary sigmoid: use raw output score
            loss = predictions[:, 0]

        # Gradient of prediction w.r.t. conv feature maps
        try:
            grads = tape.gradient(loss, conv_outputs)  # (1, H', W', C_conv)
        except Exception as e:
            logger.warning("GradientTape.gradient() failed: %s", e)
            # Return a blank heatmap instead of crashing
            dummy_shape = conv_outputs.shape[1:3]
            return np.zeros(dummy_shape, dtype=np.float32)

        if grads is None:
            logger.warning("Gradients are None — returning blank heatmap")
            dummy_shape = conv_outputs.shape[1:3]
            return np.zeros(dummy_shape, dtype=np.float32)

        # Global Average Pool the gradients → importance weights per channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C_conv,)

        conv_outputs = conv_outputs[0]  # (H', W', C_conv)

        # Weighted combination of feature maps
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H', W', 1)
        heatmap = tf.squeeze(heatmap)  # (H', W')

        # ReLU (keep only positive contributions)
        heatmap = tf.nn.relu(heatmap)

        # Normalise to [0, 1]
        heatmap = heatmap.numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + eps)

        return heatmap.astype(np.float32)


    def overlay_on_spectrogram(
        self,
        spec: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.6,
        colormap: str = "jet",
    ) -> np.ndarray:
        """
        Resize heatmap to spectrogram dimensions and create coloured overlay.

        Args:
            spec:     Original spectrogram, shape (H, W) or (H, W, 1).
            heatmap:  Raw Grad-CAM heatmap (H', W').
            alpha:    Heatmap opacity in overlay [0, 1].
            colormap: Matplotlib colourmap for heatmap.

        Returns:
            RGB overlay, shape (H, W, 3), dtype float32 ∈ [0, 1].
        """
        import matplotlib.pyplot as plt
        from PIL import Image

        if spec.ndim == 3:
            spec = spec.squeeze(-1)

        H, W = spec.shape

        # Normalise spectrogram to [0, 1] for display
        spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)

        # Resize heatmap to spectrogram size using bilinear interpolation
        heatmap_resized = np.array(
            Image.fromarray(heatmap).resize((W, H), resample=Image.BILINEAR)
        )

        # Colour the heatmap using matplotlib colourmap
        cmap   = plt.get_cmap(colormap)
        heatmap_rgb = cmap(heatmap_resized)[..., :3]  # Drop alpha → (H, W, 3)

        # Convert spectrogram to 3-channel grey
        spec_rgb = np.stack([spec_norm] * 3, axis=-1)

        # Alpha blend
        overlay = (1 - alpha) * spec_rgb + alpha * heatmap_rgb
        overlay = np.clip(overlay, 0.0, 1.0)

        return overlay.astype(np.float32)

    def explain(
        self,
        spec_input: np.ndarray,
        return_overlay: bool = True,
        alpha: float = 0.6,
    ) -> Dict:
        """
        Full Grad-CAM explanation pipeline.

        Args:
            spec_input:     Preprocessed spectrogram (H, W, 1) or (1, H, W, 1).
            return_overlay: Whether to compute the RGB overlay.
            alpha:          Heatmap opacity.

        Returns:
            Dict with:
              'heatmap':    Raw heatmap (H', W')
              'overlay':    Coloured overlay (H, W, 3) — if return_overlay
              'layer_name': Target layer name
        """
        heatmap = self.compute_heatmap(spec_input)
        result  = {
            "heatmap":    heatmap,
            "layer_name": self.layer_name,
        }

        if return_overlay:
            spec_2d = spec_input.squeeze() if spec_input.ndim > 2 else spec_input
            if spec_2d.ndim == 3:
                spec_2d = spec_2d[0]  # Remove batch dim
            result["overlay"] = self.overlay_on_spectrogram(spec_2d, heatmap, alpha=alpha)

        return result


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def get_gradcam_for_model(
    model: keras.Model,
    model_name: str,
) -> GradCAM:
    """
    Create a GradCAM instance for a named model, using the correct target layer.

    Args:
        model:      Loaded Keras model.
        model_name: 'custom_cnn', 'cnn_lstm', or 'efficientnet_b0'.

    Returns:
        Configured GradCAM instance.
    """
    layer_name = GRADCAM_LAYER_MAP.get(model_name)
    if layer_name:
        # Verify the layer exists (may differ after fine-tuning)
        layer_names = [l.name for l in model.layers]
        if layer_name not in layer_names:
            logger.warning(
                "Layer '%s' not found. Auto-detecting last Conv2D.", layer_name
            )
            layer_name = None

    return GradCAM(model, layer_name=layer_name, model_name=model_name)


# ─────────────────────────────────────────────────────────────────────────────
# BATCH EXPLANATION FOR NOTEBOOKS
# ─────────────────────────────────────────────────────────────────────────────

def explain_batch(
    gradcam: GradCAM,
    file_paths: list,
    labels: list,
    n_samples: int = 4,
    save_dir=None,
) -> None:
    """
    Generate Grad-CAM visualisations for a batch of files and save to disk.

    Args:
        gradcam:   GradCAM instance.
        file_paths: List of audio file paths.
        labels:    Corresponding integer labels (1=real, 0=spoof).
        n_samples: Number of samples per class to visualise.
        save_dir:  Directory to save PNG figures.
    """
    import matplotlib.pyplot as plt

    if save_dir:
        from pathlib import Path
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    real_paths  = [p for p, l in zip(file_paths, labels) if l == 1][:n_samples]
    spoof_paths = [p for p, l in zip(file_paths, labels) if l == 0][:n_samples]

    for label_str, paths in [("real_bonafide", real_paths), ("spoof_fake", spoof_paths)]:
        for i, path in enumerate(paths):
            try:
                spec = preprocess_file(path, add_channel_dim=True)
                explanation = gradcam.explain(spec)

                fig, axes = plt.subplots(1, 2, figsize=(14, 4))
                # Original spectrogram
                from src.utils import plot_spectrogram
                plot_spectrogram(spec, title=f"Log-Mel — {label_str}", ax=axes[0])
                # Grad-CAM overlay
                axes[1].imshow(
                    np.flipud(explanation["overlay"]),
                    aspect="auto", origin="lower",
                )
                axes[1].set_title(f"Grad-CAM Overlay — {label_str} [{gradcam.layer_name}]",
                                   fontweight="bold")
                axes[1].set_xlabel("Time Frames")
                axes[1].set_ylabel("Mel Frequency Bins")

                plt.tight_layout()
                if save_dir:
                    fname = save_dir / f"gradcam_{label_str}_{i:02d}.png"
                    fig.savefig(fname, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                else:
                    plt.show()
            except Exception as e:
                logger.warning("GradCAM failed for %s: %s", path, e)
