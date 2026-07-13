"""
DeepShield Audio — Grad-CAM View Component
===========================================
Renders the Grad-CAM heatmap overlay on the Log-Mel spectrogram.
"""

import io
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable



def render_gradcam(
    gradcam_result: dict,
    spec: np.ndarray,
    title: str = "Grad-CAM Explainability",
) -> None:
    """
    Render Grad-CAM heatmap overlay in Streamlit.

    Args:
        gradcam_result: Dict from GradCAM.explain() with 'heatmap' and 'overlay'.
        spec:           Original spectrogram (N_MELS, T_FRAMES) or (N_MELS, T_FRAMES, 1).
        title:          Section title.
    """
    if gradcam_result is None:
        st.info("ℹ️ Grad-CAM not available for this inference result.")
        return

    st.markdown(f"### 🔥 {title}")
    st.markdown(
        "_The heatmap highlights **which time-frequency regions** the model "
        "found most discriminative for its decision. "
        "**Red/warm regions** had the highest activation for the predicted class._"
    )

    overlay = gradcam_result.get("overlay")
    heatmap = gradcam_result.get("heatmap")
    layer   = gradcam_result.get("layer_name", "unknown")

    if spec.ndim == 3:
        spec_2d = spec.squeeze(-1)
    else:
        spec_2d = spec

    # ── Side-by-Side: Original + Overlay ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor("#0E1117")

    for ax in axes:
        ax.set_facecolor("#0E1117")

    # Left: original spectrogram
    axes[0].imshow(
        np.flipud(spec_2d),
        aspect="auto",
        cmap="magma",
        origin="lower",
        extent=[0, spec_2d.shape[1], 0, spec_2d.shape[0]],
    )
    axes[0].set_title("Original Log-Mel Spectrogram", color="white",
                       fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Time Frames", color="#BBBBBB", fontsize=8)
    axes[0].set_ylabel("Mel Frequency Bins", color="#BBBBBB", fontsize=8)
    axes[0].tick_params(colors="#BBBBBB", labelsize=7)
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#333333")

    # Right: Grad-CAM overlay
    if overlay is not None:
        axes[1].imshow(
            np.flipud(overlay),
            aspect="auto",
            origin="lower",
            extent=[0, overlay.shape[1], 0, overlay.shape[0]],
        )
        axes[1].set_title(
            f"Grad-CAM Overlay [{layer}]",
            color="white", fontsize=10, fontweight="bold",
        )
    elif heatmap is not None:
        axes[1].imshow(
            np.flipud(heatmap),
            aspect="auto",
            cmap="jet",
            origin="lower",
        )
        axes[1].set_title(f"Grad-CAM Heatmap [{layer}]",
                           color="white", fontsize=10, fontweight="bold")
    else:
        axes[1].text(0.5, 0.5, "Grad-CAM Not Available",
                     ha="center", va="center", color="white", fontsize=12)

    axes[1].set_xlabel("Time Frames", color="#BBBBBB", fontsize=8)
    axes[1].set_ylabel("Mel Frequency Bins", color="#BBBBBB", fontsize=8)
    axes[1].tick_params(colors="#BBBBBB", labelsize=7)
    for spine in axes[1].spines.values():
        spine.set_edgecolor("#333333")

    # Colourbar for Grad-CAM
    if overlay is not None or heatmap is not None:
        sm = ScalarMappable(cmap="jet", norm=Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[1], fraction=0.04, pad=0.02)
        cbar.set_label("Activation Intensity", color="#BBBBBB", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="#BBBBBB")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#BBBBBB", fontsize=7)

    plt.suptitle(
        "Model Explainability (Grad-CAM)",
        color="white", fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    st.image(buf, width="stretch")
    plt.close(fig)


    # ── Interpretation Guide ─────────────────────────────────────────────────
    with st.expander("ℹ️ How to interpret Grad-CAM"):
        st.markdown("""
        **Grad-CAM (Gradient-weighted Class Activation Mapping)** shows which
        regions of the spectrogram the model focused on when making its decision.

        | Colour | Meaning |
        |--------|---------|
        | 🔴 Red / Orange | High activation — model focused here |
        | 🟡 Yellow       | Moderate activation |
        | 🔵 Blue         | Low activation — less important |

        **Interpretation tips:**
        - For **fake audio**, warm regions often appear in high-frequency bands where
          TTS/VC artefacts typically manifest.
        - For **real audio**, activations tend to be more distributed across the
          natural harmonic structure.
        - Sparse, noisy activation patterns may indicate model uncertainty.
        """)
