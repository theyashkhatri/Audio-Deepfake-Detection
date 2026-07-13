"""
DeepShield Audio — Audio Player Component
==========================================
Renders waveform visualisation in the Streamlit sidebar/main panel.
"""

import io
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa


def render_audio_player(
    audio_bytes: bytes,
    waveform: np.ndarray,
    sample_rate: int,
    filename: str = "audio",
) -> None:
    """
    Render the audio player with native Streamlit audio + waveform plot.

    Args:
        audio_bytes: Raw audio bytes for the player widget.
        waveform:    1-D float32 numpy waveform.
        sample_rate: Sample rate (Hz).
        filename:    Display name.
    """
    st.markdown("### 🎵 Audio Player")
    st.audio(audio_bytes, format="audio/wav")

    # ── Waveform Plot ────────────────────────────────────────────────────────
    duration = len(waveform) / sample_rate
    time_axis = np.linspace(0, duration, num=len(waveform))

    fig, ax = plt.subplots(figsize=(10, 2.2))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    ax.plot(time_axis, waveform, linewidth=0.5, color="#4ECDC4", alpha=0.9)
    ax.fill_between(time_axis, waveform, 0, alpha=0.15, color="#4ECDC4")
    ax.axhline(0, color="#888", linewidth=0.5, linestyle="--")

    ax.set_xlabel("Time (s)", color="#BBBBBB", fontsize=9)
    ax.set_ylabel("Amplitude", color="#BBBBBB", fontsize=9)
    ax.tick_params(colors="#BBBBBB", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    ax.set_xlim([0, duration])
    ax.set_title(f"Waveform — {filename}", color="white", fontsize=10, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)  # noqa: deprecated arg OK for older streamlit
    plt.close(fig)

    # ── Stats Row ────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duration",   f"{duration:.2f} s")
    col2.metric("Sample Rate", f"{sample_rate / 1000:.0f} kHz")
    col3.metric("Samples",    f"{len(waveform):,}")
    col4.metric("Peak Level", f"{np.abs(waveform).max():.3f}")
