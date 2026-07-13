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
            <h1 style="color: {verdict_color} !important; margin: 0; font-size: 2rem; font-weight: 800;">
                {verdict_icon} {verdict_text}
            </h1>
            <p style="color: #BBBBCC !important; margin-top: 8px; font-size: 0.95rem;">
                Confidence: <strong style="color: #FFFFFF !important;">{confidence * 100:.1f}%</strong>
                &nbsp;|&nbsp;
                Threshold: <strong style="color: #FFFFFF !important;">{threshold:.2f}</strong>
                &nbsp;|&nbsp;
                Model: <strong style="color: #FFFFFF !important;">{model_name}</strong>
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
            <strong style="color: {risk_color} !important; font-size: 1.1rem; margin-left: 8px;">
                Risk Level: {risk_level}
            </strong>
            <p style="color: #BBBBCC !important; margin: 4px 0 0 0; font-size: 0.9rem;">{risk_desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Probability Breakdown ─────────────────────────────────────────────────
    st.markdown("### 📊 Probability Analysis")
    
    real_pct = real_prob * 100
    fake_pct = fake_prob * 100
    
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; gap: 12px; margin-top: 15px; margin-bottom: 20px;">
            <div style="background: #141721; padding: 12px 16px; border-radius: 10px; flex: 1; border: 1px solid #1E2330;">
                <div style="font-size: 0.65rem; color: #888899 !important; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Real Speech</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #2A9D8F !important; margin-top: 4px;">{real_pct:.2f}%</div>
                <div style="background: #222530; border-radius: 4px; height: 6px; margin-top: 8px; overflow: hidden;">
                    <div style="background: #2A9D8F; width: {real_pct}%; height: 100%;"></div>
                </div>
            </div>
            <div style="background: #141721; padding: 12px 16px; border-radius: 10px; flex: 1; border: 1px solid #1E2330;">
                <div style="font-size: 0.65rem; color: #888899 !important; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">AI Generated / Spoof</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #E63946 !important; margin-top: 4px;">{fake_pct:.2f}%</div>
                <div style="background: #222530; border-radius: 4px; height: 6px; margin-top: 8px; overflow: hidden;">
                    <div style="background: #E63946; width: {fake_pct}%; height: 100%;"></div>
                </div>
            </div>
            <div style="background: #141721; padding: 12px 16px; border-radius: 10px; flex: 1; border: 1px solid #1E2330;">
                <div style="font-size: 0.65rem; color: #888899 !important; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Latency</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #00F2FE !important; margin-top: 4px;">{infer_ms:.1f} ms</div>
                <div style="font-size: 0.65rem; color: #666677 !important; margin-top: 8px; font-weight: 500;">Threshold: {threshold:.3f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

