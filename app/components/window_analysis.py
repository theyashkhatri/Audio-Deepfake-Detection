"""
DeepShield Audio — Window Analysis Component
=============================================
Renders the per-window fake probability timeline for long audio.
"""

from typing import List, Dict

import io
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import plotly.graph_objects as go


def render_window_analysis(
    window_results: List[Dict],
    threshold: float = 0.5,
    use_plotly: bool = True,
) -> None:
    """
    Render the window-based deepfake probability timeline.

    Args:
        window_results: List of per-window dicts from predict_long_audio().
        threshold:      Decision threshold line.
        use_plotly:     Use Plotly (interactive) vs Matplotlib (static).
    """
    if not window_results:
        st.info("ℹ️ No window analysis available (audio may be too short).")
        return

    st.markdown("### 📊 Window-Based Analysis Timeline")

    starts     = [w["start_sec"] for w in window_results]
    ends       = [w["end_sec"]   for w in window_results]
    fake_probs = [w["fake_prob"] for w in window_results]
    real_probs = [w["real_prob"] for w in window_results]
    labels     = [w["label"]     for w in window_results]

    n_fake = sum(1 for l in labels if l == "spoof")
    n_real = sum(1 for l in labels if l == "bonafide")

    # ── Summary Metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Windows",     len(window_results))
    col2.metric("🔴 Fake Windows",   n_fake)
    col3.metric("🟢 Real Windows",   n_real)
    col4.metric("Avg P(Fake)",        f"{np.mean(fake_probs):.3f}")

    # ── Interactive Plotly Chart ──────────────────────────────────────────────
    if use_plotly:
        _render_plotly_timeline(starts, ends, fake_probs, labels, threshold)
    else:
        _render_matplotlib_timeline(window_results, threshold)

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
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="white"),
        showlegend=False,
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(gridcolor="#2A2A2A", linecolor="#444")
    fig.update_yaxes(gridcolor="#2A2A2A", linecolor="#444")

    st.plotly_chart(fig, width="stretch")


def _render_matplotlib_timeline(window_results: list, threshold: float) -> None:
    """Fallback matplotlib chart."""
    starts     = [w["start_sec"] for w in window_results]
    fake_probs = [w["fake_prob"] for w in window_results]
    widths     = [w["end_sec"] - w["start_sec"] for w in window_results]
    colors     = ["#E63946" if w["is_fake"] else "#2A9D8F" for w in window_results]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    ax.bar(starts, fake_probs, width=[w * 0.85 for w in widths],
           align="edge", color=colors, edgecolor="#444", linewidth=0.5)

    ax.axhline(threshold, color="white", linewidth=1.5, linestyle="--",
               label=f"Threshold ({threshold:.2f})")

    ax.set_xlabel("Time (s)", color="#BBBBBB")
    ax.set_ylabel("P(Fake)", color="#BBBBBB")
    ax.set_title("Window Analysis", color="white", fontweight="bold")
    ax.tick_params(colors="#BBBBBB")
    ax.legend(facecolor="#1E2028", edgecolor="#444", labelcolor="white")
    ax.set_ylim([0, 1.05])
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    st.image(buf, width="stretch")
    plt.close(fig)

