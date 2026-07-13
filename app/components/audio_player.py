"""
DeepShield Audio — Audio Player Component
==========================================
Renders waveform visualisation in the Streamlit sidebar/main panel.
Uses st.image() instead of st.pyplot() to avoid matplotlib/Streamlit
segfault on Apple Silicon (M1/M2/M3).
"""

import io
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

    # Render to buffer → st.image() avoids Streamlit/matplotlib segfault on Apple Silicon
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    st.image(buf, width="stretch")
    plt.close(fig)

    # ── Stats Row ────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; gap: 8px; margin-top: 15px;">
            <div style="background: #141721; padding: 10px; border-radius: 8px; flex: 1; text-align: center; border: 1px solid #1E2330;">
                <div style="font-size: 0.65rem; color: #888899 !important; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Duration</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #00F2FE !important; margin-top: 4px;">{duration:.2f} s</div>
            </div>
            <div style="background: #141721; padding: 10px; border-radius: 8px; flex: 1; text-align: center; border: 1px solid #1E2330;">
                <div style="font-size: 0.65rem; color: #888899 !important; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Sample Rate</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #00F2FE !important; margin-top: 4px;">{sample_rate / 1000:.0f} kHz</div>
            </div>
            <div style="background: #141721; padding: 10px; border-radius: 8px; flex: 1; text-align: center; border: 1px solid #1E2330;">
                <div style="font-size: 0.65rem; color: #888899 !important; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Samples</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #00F2FE !important; margin-top: 4px;">{len(waveform):,}</div>
            </div>
            <div style="background: #141721; padding: 10px; border-radius: 8px; flex: 1; text-align: center; border: 1px solid #1E2330;">
                <div style="font-size: 0.65rem; color: #888899 !important; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Peak Level</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #00F2FE !important; margin-top: 4px;">{np.abs(waveform).max():.3f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

