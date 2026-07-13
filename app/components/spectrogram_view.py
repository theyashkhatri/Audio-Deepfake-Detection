"""
DeepShield Audio — Spectrogram View Component
===============================================
Renders Log-Mel spectrogram with frequency axis and colour bar.
Uses st.image() instead of st.pyplot() to avoid matplotlib/Streamlit
segfault on Apple Silicon (M1/M2/M3).
"""

import io
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display


def render_spectrogram(
    spec: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 256,
    title: str = "Log-Mel Spectrogram",
    cmap: str = "magma",
) -> None:
    """
    Render a Log-Mel spectrogram in Streamlit.

    Args:
        spec:        (N_MELS, T_FRAMES) or (N_MELS, T_FRAMES, 1) array in dB.
        sample_rate: Audio sample rate.
        hop_length:  STFT hop length for time axis.
        title:       Plot title.
        cmap:        Matplotlib colourmap.
    """
    st.markdown(f"### 🎨 {title}")

    if spec.ndim == 3:
        spec = spec.squeeze(-1)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    img = librosa.display.specshow(
        spec,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=ax,
        cmap=cmap,
    )
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.outline.set_edgecolor("white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
    cbar.set_label("dB", color="white", fontsize=9)

    ax.set_title(title, color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)", color="#BBBBBB", fontsize=9)
    ax.set_ylabel("Mel Frequency (Hz)", color="#BBBBBB", fontsize=9)
    ax.tick_params(colors="#BBBBBB", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    plt.tight_layout()

    # Render to buffer → st.image() avoids Streamlit/matplotlib segfault on Apple Silicon
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    st.image(buf, width="stretch")
    plt.close(fig)
