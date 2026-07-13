"""
DeepShield Audio — Grad-CAM View Component
===========================================
Renders the Grad-CAM heatmap overlay on the Log-Mel spectrogram.
"""

import numpy as np
import streamlit as st


def render_gradcam(
    gradcam_result: dict,
    spec: np.ndarray,
    title: str = "Grad-CAM Explainability",
) -> None:
    """
    Render Grad-CAM heatmap overlay in Streamlit using native st.image (librosa/matplotlib-free).

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
    layer   = gradcam_result.get("layer_name", "unknown")

    if overlay is not None:
        overlay_arr = np.array(overlay)
        # Flip vertically to match spectrogram orientation
        overlay_flipped = np.flipud(overlay_arr)
        
        # Ensure correct uint8 format for st.image
        if overlay_flipped.dtype != np.uint8:
            overlay_flipped = np.clip(overlay_flipped, 0, 255).astype(np.uint8)

        st.markdown(f"**Grad-CAM Superimposed Overlay [Layer: `{layer}`]**")
        st.image(overlay_flipped, use_container_width=True)
    else:
        st.warning("Grad-CAM visualization array is not available.")

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

