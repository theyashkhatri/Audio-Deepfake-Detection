"""
DeepShield Audio — Window Analysis Component
=============================================
Renders the per-window fake probability timeline for long audio.
"""

from typing import List, Dict
import numpy as np
import streamlit as st
import plotly.graph_objects as go


def render_window_analysis(
    window_results: List[Dict],
    threshold: float = 0.5,
) -> None:
    """
    Render the window-based deepfake probability timeline using Plotly.

    Args:
        window_results: List of per-window dicts from predict_long_audio().
        threshold:      Decision threshold line.
    """
    if not window_results:
        st.info("ℹ️ No window analysis available (audio may be too short).")
        return

    st.markdown("### 📊 Window-Based Analysis Timeline")

    starts     = [w["start_sec"] for w in window_results]
    ends       = [w["end_sec"]   for w in window_results]
    fake_probs = [w["fake_prob"] for w in window_results]
    labels     = [w["label"]     for w in window_results]

    n_fake = sum(1 for l in labels if l == "spoof")
    n_real = sum(1 for l in labels if l == "bonafide")

    # ── Summary Metrics ───────────────────────────────────────────────────────
    avg_fake = np.mean(fake_probs)
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; gap: 8px; margin-top: 10px; margin-bottom: 20px;">
            <div style="background: #141721; padding: 10px; border-radius: 8px; flex: 1; text-align: center; border: 1px solid #1E2330;">
                <div style="font-size: 0.65rem; color: #888899 !important; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Total Windows</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #00F2FE !important; margin-top: 4px;">{len(window_results)}</div>
            </div>
            <div style="background: #141721; padding: 10px; border-radius: 8px; flex: 1; text-align: center; border: 1px solid #1E2330;">
                <div style="font-size: 0.65rem; color: #888899 !important; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">🔴 Fake Windows</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #E63946 !important; margin-top: 4px;">{n_fake}</div>
            </div>
            <div style="background: #141721; padding: 10px; border-radius: 8px; flex: 1; text-align: center; border: 1px solid #1E2330;">
                <div style="font-size: 0.65rem; color: #888899 !important; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">🟢 Real Windows</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #2A9D8F !important; margin-top: 4px;">{n_real}</div>
            </div>
            <div style="background: #141721; padding: 10px; border-radius: 8px; flex: 1; text-align: center; border: 1px solid #1E2330;">
                <div style="font-size: 0.65rem; color: #888899 !important; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Avg P(Fake)</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #00F2FE !important; margin-top: 4px;">{avg_fake:.3f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Interactive Plotly Chart ──────────────────────────────────────────────
    _render_plotly_timeline(starts, ends, fake_probs, labels, threshold)

    # ── Window Table ──────────────────────────────────────────────────────────
    with st.expander("📋 Window-by-Window Details", expanded=False):
        import pandas as pd
        rows = []
        for w in window_results:
            rows.append({
                "Window":    w["window_idx"],
                "Start (s)": f"{w['start_sec']:.2f}",
                "End (s)":   f"{w['end_sec']:.2f}",
                "P(Real)":   f"{w['real_prob']:.4f}",
                "P(Fake)":   f"{w['fake_prob']:.4f}",
                "Verdict":   "🔴 FAKE" if w["is_fake"] else "🟢 REAL",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch")


def _render_plotly_timeline(
    starts: list,
    ends: list,
    fake_probs: list,
    labels: list,
    threshold: float,
) -> None:
    """Plotly interactive bar chart."""
    colors = ["#E63946" if l == "spoof" else "#2A9D8F" for l in labels]
    widths = [e - s for s, e in zip(starts, ends)]

    fig = go.Figure()

    # Bar trace
    fig.add_trace(go.Bar(
        x=[s + w / 2 for s, w in zip(starts, widths)],
        y=fake_probs,
        width=widths,
        marker_color=colors,
        marker_line_color="rgba(255,255,255,0.3)",
        marker_line_width=0.5,
        name="P(Fake)",
        hovertemplate=(
            "<b>Window start:</b> %{customdata[0]:.2f}s<br>"
            "<b>P(Fake):</b> %{y:.4f}<br>"
            "<b>Verdict:</b> %{customdata[1]}<extra></extra>"
        ),
        customdata=[[s, l.upper()] for s, l in zip(starts, labels)],
    ))

    # Threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="white",
        line_width=1.5,
        annotation_text=f"Threshold ({threshold:.2f})",
        annotation_font_color="white",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Fake Probability per Window",
        xaxis_title="Time (seconds)",
        yaxis_title="P(Fake)",
        yaxis_range=[0, 1.05],
        plot_bgcolor="#0E1116",
        paper_bgcolor="#0E1116",
        font=dict(color="white"),
        showlegend=False,
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(gridcolor="#2A2A2A", linecolor="#444")
    fig.update_yaxes(gridcolor="#2A2A2A", linecolor="#444")
    st.plotly_chart(fig, width="stretch")

