"""
DeepShield Audio — Main Streamlit Application
==============================================
Professional AI audio deepfake detection system with:
  - WAV / MP3 / FLAC upload
  - Waveform + Log-Mel spectrogram visualisation
  - AI prediction (Real vs Fake) with confidence
  - Risk band assessment
  - Window-based long audio analysis
  - Grad-CAM explainability

Run with:
    streamlit run app/app.py
"""


import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# ── Path Setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import streamlit as st




# ── Streamlit Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepShield Audio — AI Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/deepshield-audio",
        "Report a bug": "https://github.com/yourusername/deepshield-audio/issues",
        "About": "DeepShield Audio v1.0 — AI-Generated Voice Detection",
    },
)

# ── Custom CSS (Dark Premium Theme) ──────────────────────────────────────────
st.markdown("""
<style>
    /* Global dark background & high-contrast font styling */
    .stApp { background-color: #0A0C10; }
    .main .block-container { padding: 2rem 3rem; max-width: 1400px; }

    /* Base high-contrast body text */
    div, p, span, label, li, ul, ol, td, th {
        color: #E2E8F0 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Bright white headers */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }

    /* Sidebar text high-contrast */
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] span {
        color: #E2E8F0 !important;
    }

    /* Sidebar panel clean layout */
    [data-testid="stSidebar"] {
        background: #0E1116;
        border-right: 1px solid #1E2330;
    }

    /* Premium card elements */
    .premium-card {
        background: #0E1116;
        border: 1px solid #1E2330;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        margin-bottom: 24px;
    }

    /* Header brand - minimal */
    .brand-header {
        border-bottom: 1px solid #1E2330;
        padding-bottom: 16px;
        margin-bottom: 32px;
        text-align: left;
    }
    .brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #00F2FE, #4FACFE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        color: unset !important; /* Allow background gradient clip */
    }
    .brand-subtitle {
        color: #888899 !important;
        font-size: 0.95rem;
        margin-top: 4px;
    }

    /* Upload zone - clean minimal */
    [data-testid="stFileUploader"] {
        border: 1px dashed #1E2330 !important;
        border-radius: 12px !important;
        background: #0E1116 !important;
        padding: 10px !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: #0E1116 !important;
    }

    /* Section divider */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #00F2FE !important;
        border-bottom: 1px solid #1E2330;
        padding-bottom: 6px;
        margin-bottom: 20px;
    }

    /* Spinner override */
    .stSpinner > div { border-top-color: #00F2FE !important; }

    /* Progress bars */
    .stProgress > div > div { background: linear-gradient(90deg, #4FACFE, #00F2FE) !important; }

    /* Expander text */
    .streamlit-expanderHeader {
        background: #0E1116 !important;
        border: 1px solid #1E2330 !important;
        border-radius: 8px !important;
    }
    .streamlit-expanderHeader p {
        color: #FFFFFF !important;
    }

    /* Download button styling */
    [data-testid="stDownloadButton"] button {
        background: linear-gradient(90deg, #4FACFE, #00F2FE) !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def check_model_exists(model_name: str):
    from src.config import SAVED_MODELS_DIR
    model_path = SAVED_MODELS_DIR / model_name / f"{model_name}_best.keras"
    if not model_path.exists():
        return False, f"Model file not found: {model_path}"
    return True, None



# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 20px 0;">
            <div style="font-size: 2.5rem;">🛡️</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #4ECDC4;">DeepShield Audio</div>
            <div style="font-size: 0.75rem; color: #666;">v1.0 — AI Deepfake Detector</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Model Selection ───────────────────────────────────────────────────────
    st.markdown("#### 🧠 Model Selection")
    model_options = {
        "Custom CNN":        "custom_cnn",
        "CNN-LSTM Hybrid":   "cnn_lstm",
        "EfficientNetB0":    "efficientnet_b0",
    }
    selected_model_label = st.selectbox(
        "Choose detection model:",
        list(model_options.keys()),
        index=0,
        help="Each model offers different accuracy/speed trade-offs.",
    )
    selected_model_name = model_options[selected_model_label]

    # ── Threshold Control ─────────────────────────────────────────────────────
    st.markdown("#### 🎯 Decision Threshold")
    threshold = st.slider(
        "Fake probability threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.01,
        help="Probability above this → classified as FAKE. "
             "Lower = more sensitive (catches more fakes, higher false positives).",
    )

    # ── Analysis Mode ─────────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Analysis Settings")
    use_windowed = st.checkbox(
        "Window-based analysis",
        value=True,
        help="Break long audio into overlapping windows for granular analysis.",
    )
    window_sec = st.slider(
        "Window size (s)", 1.0, 5.0, 3.0, 0.5,
        disabled=not use_windowed,
        help="Duration of each analysis window.",
    )
    hop_sec = st.slider(
        "Window hop (s)", 0.5, 3.0, 1.5, 0.5,
        disabled=not use_windowed,
        help="Step between windows (< window size = overlap).",
    )

    show_gradcam = st.checkbox(
        "🔥 Show Grad-CAM explanation",
        value=False,
        help="Generate explainability heatmap. ⚠️ May crash on Apple Silicon (M1/M2/M3) due to a TensorFlow GradientTape bug — disable if app crashes after upload.",
    )

    st.divider()

    # ── Info Box ──────────────────────────────────────────────────────────────
    with st.expander("ℹ️ About DeepShield", expanded=False):
        st.markdown("""
        **DeepShield Audio** uses deep learning to detect AI-generated
        and spoofed speech trained on the ASVspoof 2019 LA dataset.

        **Models:**
        - 🏗️ **Custom CNN** — Fast 4-block CNN
        - 🔄 **CNN-LSTM** — Temporal modelling
        - 🚀 **EfficientNetB0** — Transfer learning

        **Supported formats:** WAV, MP3, FLAC

        **Dataset:** [ASVspoof 2019 LA](https://datashare.ed.ac.uk/handle/10283/3336)
        """)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="brand-header">
    <h1 class="brand-title">🛡️ DeepShield Audio</h1>
    <p class="brand-subtitle">
        AI-Generated Voice & Audio Deepfake Detection System
        &nbsp;•&nbsp; Powered by Deep Learning &nbsp;•&nbsp; ASVspoof 2019 Trained
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

is_trained, model_error = check_model_exists(selected_model_name)

if not is_trained:
    st.warning(
        f"⚠️ **Model not found:** {model_error}\n\n"
        "Running in **demo mode** with random weights. "
        "Train the models first using `notebooks/03_Model_Training.ipynb`.",
    )
    st.info("🎭 **Demo mode active** — using untrained model (random predictions).")



# ─────────────────────────────────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">📂 Upload Audio File</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Drag and drop or click to upload",
    type=["wav", "mp3", "flac"],
    help="Supported formats: WAV, MP3, FLAC. Max size: 200 MB.",
    label_visibility="collapsed",
)

if uploaded_file is None:
    # ── Landing State ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        text-align: center;
        padding: 60px 20px;
        background: #0E1116;
        border: 1px dashed #1E2330;
        border-radius: 14px;
        margin-top: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    ">
        <div style="font-size: 4rem; margin-bottom: 16px;">🎙️</div>
        <h3 style="color: #FFFFFF !important; font-weight: 700;">Upload an audio file to begin analysis</h3>
        <p style="color: #888899 !important; margin-top: 8px; font-size: 0.95rem;">
            Accepts WAV · MP3 · FLAC &nbsp;|&nbsp; Any duration
        </p>
        <div style="display: flex; justify-content: center; gap: 30px; margin-top: 30px;">
            <div style="text-align: center; color: #E2E8F0 !important;">
                <div style="font-size: 1.8rem;">🔍</div>
                <div style="font-size: 0.8rem; margin-top: 6px; font-weight: 500;">AI Detection</div>
            </div>
            <div style="text-align: center; color: #E2E8F0 !important;">
                <div style="font-size: 1.8rem;">📊</div>
                <div style="font-size: 0.8rem; margin-top: 6px; font-weight: 500;">Window Analysis</div>
            </div>
            <div style="text-align: center; color: #E2E8F0 !important;">
                <div style="font-size: 1.8rem;">🔥</div>
                <div style="font-size: 0.8rem; margin-top: 6px; font-weight: 500;">Grad-CAM XAI</div>
            </div>
            <div style="text-align: center; color: #E2E8F0 !important;">
                <div style="font-size: 1.8rem;">📥</div>
                <div style="font-size: 0.8rem; margin-top: 6px; font-weight: 500;">JSON Report</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS UPLOADED FILE
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("🔍 Analysing audio …"):
    # Save to temp file for librosa to read
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        import subprocess
        import json
        from src.config import SAMPLE_RATE

        t0 = time.perf_counter()

        # Build CLI command to isolate TensorFlow execution in its own process
        cmd = [
            ".venv/bin/python",
            "-m",
            "src.inference",
            tmp_path,
            "--model",
            selected_model_name,
            "--threshold",
            str(threshold),
            "--json"
        ]
        if use_windowed:
            cmd.append("--windowed")
            cmd.extend(["--window-sec", str(window_sec)])
            cmd.extend(["--hop-sec", str(hop_sec)])
        if show_gradcam:
            cmd.append("--gradcam")

        # Run process with PYTHONPATH set so src packages import correctly
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "."}
        )

        if res.returncode != 0:
            st.error(f"❌ Inference process crashed: {res.stderr or res.stdout}")
            st.stop()

        try:
            result = json.loads(res.stdout)
        except Exception as e:
            st.error(f"❌ Failed to parse inference output: {e}\nRaw output: {res.stdout}")
            st.stop()

        # Reconstruct NumPy arrays safely from JSON fields
        waveform = np.array(result.get("waveform", []), dtype=np.float32)
        spec = np.array(result.get("spec", []), dtype=np.float32)

        single_result = result

        # Reconstruct NumPy arrays for Grad-CAM overlay if present in JSON response
        if "gradcam" in single_result and single_result["gradcam"] is not None:
            g = single_result["gradcam"]
            if "heatmap" in g:
                g["heatmap"] = np.array(g["heatmap"], dtype=np.float32)
            if "overlay" in g:
                g["overlay"] = np.array(g["overlay"], dtype=np.float32)

        # Inject computed spec and waveform back
        single_result["spec"] = spec
        result["spec"] = spec
        single_result["waveform"] = waveform
        result["waveform"] = waveform

        total_ms = (time.perf_counter() - t0) * 1000

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION TABS
# ─────────────────────────────────────────────────────────────────────────────

# Main two-column dashboard split
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    # 1. Prediction verdict card
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    from app.components.prediction_panel import render_prediction_panel
    render_prediction_panel(result)
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Waveform & audio player card
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    from app.components.audio_player import render_audio_player
    uploaded_file.seek(0)
    render_audio_player(
        audio_bytes=uploaded_file.read(),
        waveform=waveform,
        sample_rate=SAMPLE_RATE,
        filename=uploaded_file.name,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # 3. Log-Mel spectrogram card
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    from app.components.spectrogram_view import render_spectrogram
    spec = single_result.get("spec")
    if spec is not None:
        render_spectrogram(
            spec,
            sample_rate=SAMPLE_RATE,
            title=f"Spectrogram — {uploaded_file.name}",
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # 4. Grad-CAM card (if enabled)
    if show_gradcam:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        from app.components.gradcam_view import render_gradcam
        gradcam_result = single_result.get("gradcam")
        if spec is not None:
            render_gradcam(gradcam_result, spec)
        else:
            st.warning("Spectrogram not available for Grad-CAM.")
        st.markdown('</div>', unsafe_allow_html=True)

# 5. Full width Window analysis (if enabled)
if use_windowed and "windows" in result:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    from app.components.window_analysis import render_window_analysis
    render_window_analysis(result["windows"], threshold=threshold)
    st.markdown('</div>', unsafe_allow_html=True)

# 6. Full width JSON report card
st.markdown('<div class="premium-card">', unsafe_allow_html=True)
st.markdown("### 📥 Download Analysis Report")
report = {
    "filename":       uploaded_file.name,
    "model_used":     selected_model_name,
    "threshold":      threshold,
    "analysis_mode":  "windowed" if use_windowed else "single",
    "verdict": {
        "label":      result.get("label") or result.get("overall_label"),
        "is_fake":    result.get("is_fake") if "is_fake" in result else result.get("overall_is_fake"),
        "real_prob":  result.get("real_prob") or result.get("aggregate_real_prob"),
        "fake_prob":  result.get("fake_prob") or result.get("aggregate_fake_prob"),
        "confidence": result.get("confidence") or result.get("overall_confidence"),
    },
    "risk": result.get("risk", {}),
    "inference_ms": result.get("inference_ms", total_ms),
}
if use_windowed and "windows" in result:
    report["window_analysis"] = {
        "n_windows":     result.get("n_windows"),
        "duration_sec":  result.get("duration_sec"),
        "windows":       result.get("windows", []),
    }
report_json = json.dumps(report, indent=2, ensure_ascii=False)

col_rep_btn, col_rep_space = st.columns([1, 2])
with col_rep_btn:
    st.download_button(
        label="⬇️ Download JSON Report",
        data=report_json,
        file_name=f"deepshield_report_{Path(uploaded_file.name).stem}.json",
        mime="application/json",
    )

with st.expander("📄 View Report JSON", expanded=False):
    st.json(report)
st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f"""
    <div style="text-align: center; color: #444; font-size: 0.8rem; padding: 10px 0;">
        🛡️ <strong style="color: #4ECDC4;">DeepShield Audio</strong> v1.0
        &nbsp;|&nbsp; Model: <strong>{selected_model_name}</strong>
        &nbsp;|&nbsp; Total inference: <strong>{total_ms:.0f} ms</strong>
        &nbsp;|&nbsp; Trained on ASVspoof 2019 LA dataset
    </div>
    """,
    unsafe_allow_html=True,
)
