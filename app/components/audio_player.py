"""
DeepShield Audio — Audio Player Component
==========================================
Renders waveform visualisation in the Streamlit sidebar/main panel.
Uses Plotly for interactive, thread-safe rendering on Apple Silicon.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go


def render_audio_player(
    audio_bytes: bytes,
    waveform: np.ndarray,
    sample_rate: int,
    filename: str = "audio",
) -> None:
    """
    Render the audio player with native Streamlit audio + Plotly waveform plot.

    Args:
        audio_bytes: Raw audio bytes for the player widget.
        waveform:    1-D float32 numpy waveform.
        sample_rate: Sample rate (Hz).
        filename:    Display name.
    """
    st.markdown("### 🎵 Audio Player")
    st.audio(audio_bytes, format="audio/wav")

    # ── Waveform Plot using Plotly (Thread-Safe) ─────────────────────────────
    duration = len(waveform) / sample_rate
    
    # Downsample waveform for ultra-fast browser rendering (2000 points is plenty)
    step = max(1, len(waveform) // 2000)
    downsampled = waveform[::step]
    time_axis = np.linspace(0, duration, num=len(downsampled))

    fig = go.Figure()
    
    # Trace for the line
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=downsampled,
        mode="lines",
        line=dict(color="#00F2FE", width=1.2),
        fill="tozeroy",
        fillcolor="rgba(0, 242, 254, 0.1)",
        name="Waveform",
        hovertemplate="<b>Time:</b> %{x:.3f}s<br><b>Amplitude:</b> %{y:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text=f"Waveform — {filename}",
            font=dict(color="white", size=11, family="sans-serif", weight="bold"),
            y=0.9
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0E1116",
        plot_bgcolor="#0E1116",
        xaxis=dict(
            title=dict(text="Time (s)", font=dict(color="#BBBBBB", size=9)),
            color="#BBBBBB",
            gridcolor="#222831",
            linecolor="#333333",
            range=[0, duration]
        ),
        yaxis=dict(
            title=dict(text="Amplitude", font=dict(color="#BBBBBB", size=9)),
            color="#BBBBBB",
            gridcolor="#222831",
            linecolor="#333333",
            range=[-1.05, 1.05]
        ),
        height=180,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

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


