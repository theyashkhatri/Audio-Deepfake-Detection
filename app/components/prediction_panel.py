"""
DeepShield Audio — Prediction Panel Component
==============================================
Renders the main AI verdict: label, confidence gauge, risk badge,
and probability breakdown.
"""

import numpy as np
import streamlit as st


def render_prediction_panel(result: dict) -> None:
    """
    Render the complete prediction result panel.

    Args:
        result: Dict from AudioInferenceEngine.predict() or predict_long_audio().
                Required keys: label/overall_label, is_fake/overall_is_fake,
                real_prob/aggregate_real_prob, fake_prob/aggregate_fake_prob,
                confidence/overall_confidence, risk, inference_ms.
    """
    # Normalise keys (single-file vs long-audio)
    label      = result.get("label") or result.get("overall_label", "unknown")
    is_fake    = result.get("is_fake") if "is_fake" in result else result.get("overall_is_fake", False)
    real_prob  = result.get("real_prob") or result.get("aggregate_real_prob", 0.5)
    fake_prob  = result.get("fake_prob") or result.get("aggregate_fake_prob", 0.5)
    confidence = result.get("confidence") or result.get("overall_confidence", 0.0)
    risk       = result.get("risk", {})
    infer_ms   = result.get("inference_ms", 0.0)
    model_name = result.get("model_name", "N/A")
    threshold  = result.get("threshold", 0.5)

    risk_level  = risk.get("level", "UNKNOWN")
    risk_emoji  = risk.get("emoji", "⚪")
    risk_color  = risk.get("color", "#888")
    risk_desc   = risk.get("description", "")

    # ── Main Verdict ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🎯 AI Detection Result")

    if is_fake:
        verdict_color = "#E63946"
        verdict_icon  = "🚨"
        verdict_text  = "FAKE — AI GENERATED / SPOOFED"
    else:
        verdict_color = "#2A9D8F"
        verdict_icon  = "✅"
        verdict_text  = "REAL — BONAFIDE SPEECH"

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {verdict_color}22, {verdict_color}11);
            border: 2px solid {verdict_color};
            border-radius: 12px;
            padding: 20px 28px;
            text-align: center;
            margin-bottom: 16px;
        ">
            <h1 style="color: {verdict_color}; margin: 0; font-size: 2.2rem;">
                {verdict_icon} {verdict_text}
            </h1>
            <p style="color: #CCCCCC; margin-top: 8px; font-size: 0.95rem;">
                Confidence: <strong style="color: white;">{confidence * 100:.1f}%</strong>
                &nbsp;|&nbsp;
                Threshold: <strong style="color: white;">{threshold:.2f}</strong>
                &nbsp;|&nbsp;
                Model: <strong style="color: white;">{model_name}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Risk Band ─────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background: {risk_color}22;
            border-left: 5px solid {risk_color};
            border-radius: 8px;
            padding: 12px 20px;
            margin-bottom: 16px;
        ">
            <span style="font-size: 1.4rem;">{risk_emoji}</span>
            <strong style="color: {risk_color}; font-size: 1.1rem; margin-left: 8px;">
                Risk Level: {risk_level}
            </strong>
            <p style="color: #AAAAAA; margin: 4px 0 0 0; font-size: 0.9rem;">{risk_desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Probability Breakdown ─────────────────────────────────────────────────
    st.markdown("### 📊 Probability Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="🟢 Real (Bonafide) Probability",
            value=f"{real_prob * 100:.2f}%",
            delta=None,
        )
        st.progress(float(real_prob), text=f"{real_prob * 100:.1f}% Real")

    with col2:
        st.metric(
            label="🔴 Fake (Spoof) Probability",
            value=f"{fake_prob * 100:.2f}%",
            delta=None,
        )
        st.progress(float(fake_prob), text=f"{fake_prob * 100:.1f}% Fake")

    with col3:
        st.metric(label="⚡ Inference Time", value=f"{infer_ms:.1f} ms")
        st.metric(label="🎯 Decision Threshold", value=f"{threshold:.3f}")

    # ── Confidence Gauge (CSS progress bar) ─────────────────────────────────
    st.markdown("### 🔵 Confidence Gauge")
    gauge_color = verdict_color
    st.markdown(
        f"""
        <div style="background: #1E2028; border-radius: 8px; padding: 4px; margin-bottom: 8px;">
            <div style="
                width: {confidence * 100:.1f}%;
                background: linear-gradient(90deg, {gauge_color}88, {gauge_color});
                height: 22px;
                border-radius: 6px;
                transition: width 0.6s ease;
                display: flex;
                align-items: center;
                justify-content: flex-end;
                padding-right: 8px;
            ">
                <span style="color: white; font-size: 0.8rem; font-weight: bold;">
                    {confidence * 100:.1f}%
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
