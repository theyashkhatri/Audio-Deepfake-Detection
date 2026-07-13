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

# ── Force TensorFlow CPU-only (prevents Metal GPU segfault on Apple Silicon) ──
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")          # disable CUDA GPU
os.environ.setdefault("TF_METAL_DEVICE_ENABLE", "0")         # disable Metal GPU
os.environ.setdefault("METAL_DEVICE_WRAPPER_TYPE", "0")       # disable Metal wrapper
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")           # suppress TF logs
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import streamlit as st

# ── Path Setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


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
    /* Global dark background */
    .stApp { background-color: #0E1117; }
    .main .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12151C 0%, #0E1117 100%);
        border-right: 1px solid #2A2A35;
    }

    /* Header brand */
    .brand-header {
        background: linear-gradient(135deg, #1A1F2E, #0E1117);
        border: 1px solid #2E3250;
        border-radius: 14px;
        padding: 20px 28px;
        margin-bottom: 24px;
        text-align: center;
    }
    .brand-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4ECDC4, #457BDB, #E63946);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }
    .brand-subtitle {
        color: #888;
        font-size: 1rem;
        margin-top: 6px;
    }

    /* Upload zone */
    [data-testid="stFileUploader"] {
        border: 2px dashed #2E3250 !important;
        border-radius: 12px !important;
        background: #12151C !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #12151C;
        border: 1px solid #2A2A35;
        border-radius: 10px;
        padding: 12px;
    }

    /* Section divider */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #4ECDC4;
        border-bottom: 1px solid #2A2A35;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }

    /* Spinner override */
    .stSpinner > div { border-top-color: #4ECDC4 !important; }

    /* Progress bars */
    .stProgress > div > div { background: linear-gradient(90deg, #4ECDC4, #457BDB) !important; }

    /* Expander */
    .streamlit-expanderHeader {
        background: #12151C !important;
        border: 1px solid #2A2A35 !important;
        border-radius: 8px !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { background: #12151C; }
    .stTabs [aria-selected="true"] { border-bottom-color: #4ECDC4 !important; }

    /* Download button */
    [data-testid="stDownloadButton"] button {
        background: linear-gradient(90deg, #2A9D8F, #457BDB) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI model …")
def load_model_cached(model_name: str):
    """Load and cache a trained model (cached across Streamlit reruns)."""
    try:
        from src.trainer import load_model
        model = load_model(model_name)
        return model, None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Error loading model: {e}"


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

model_obj, model_error = load_model_cached(selected_model_name)

if model_error:
    st.warning(
        f"⚠️ **Model not found:** {model_error}\n\n"
        "Running in **demo mode** with random weights. "
        "Train the models first using `notebooks/03_Model_Training.ipynb`.",
    )
    # Load a fresh untrained model for demo
    try:
        if selected_model_name == "custom_cnn":
            from src.models.custom_cnn import build_custom_cnn
            model_obj = build_custom_cnn()
        elif selected_model_name == "cnn_lstm":
            from src.models.cnn_lstm import build_cnn_lstm
            model_obj = build_cnn_lstm()
        else:
            from src.models.efficientnet import build_efficientnet_b0
            model_obj = build_efficientnet_b0(freeze_backbone=True)
        st.info("🎭 **Demo mode active** — using untrained model (random predictions).")
    except Exception as e:
        st.error(f"❌ Failed to load any model: {e}")
        st.stop()


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
        background: #12151C;
        border: 1px dashed #2A2A35;
        border-radius: 14px;
        margin-top: 20px;
    ">
        <div style="font-size: 4rem; margin-bottom: 16px;">🎙️</div>
        <h3 style="color: #AAAAAA;">Upload an audio file to begin analysis</h3>
        <p style="color: #666; margin-top: 8px;">
            Accepts WAV · MP3 · FLAC &nbsp;|&nbsp; Any duration
        </p>
        <div style="display: flex; justify-content: center; gap: 30px; margin-top: 24px;">
            <div style="text-align: center; color: #555;">
                <div style="font-size: 1.8rem;">🔍</div>
                <div style="font-size: 0.8rem; margin-top: 4px;">AI Detection</div>
            </div>
            <div style="text-align: center; color: #555;">
                <div style="font-size: 1.8rem;">📊</div>
                <div style="font-size: 0.8rem; margin-top: 4px;">Window Analysis</div>
            </div>
            <div style="text-align: center; color: #555;">
                <div style="font-size: 1.8rem;">🔥</div>
                <div style="font-size: 0.8rem; margin-top: 4px;">Grad-CAM XAI</div>
            </div>
            <div style="text-align: center; color: #555;">
                <div style="font-size: 1.8rem;">📥</div>
                <div style="font-size: 0.8rem; margin-top: 4px;">JSON Report</div>
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
        import librosa
        from src.config import SAMPLE_RATE, WINDOW_SECONDS, HOP_SECONDS
        from src.inference import AudioInferenceEngine
        from src.preprocessor import load_audio

        # Load waveform
        waveform, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)

        # Build inference engine with selected threshold
        engine = AudioInferenceEngine(
            model_obj,
            model_name=selected_model_name,
            threshold=threshold,
        )

        t0 = time.perf_counter()

        # Run inference
        if use_windowed:
            result = engine.predict_long_audio(
                tmp_path,
                window_sec=window_sec,
                hop_sec=hop_sec,
            )
            # Also get single-file result for Grad-CAM
            single_result = engine.predict(tmp_path, return_gradcam=show_gradcam)
        else:
            result = engine.predict(tmp_path, return_gradcam=show_gradcam)
            single_result = result

        total_ms = (time.perf_counter() - t0) * 1000

    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_audio, tab_spec, tab_result, tab_windows, tab_gradcam, tab_report = st.tabs([
    "🎵 Audio",
    "🎨 Spectrogram",
    "🎯 Result",
    "📊 Window Analysis",
    "🔥 Grad-CAM",
    "📥 Report",
])

# ── Tab 1: Audio Player ───────────────────────────────────────────────────────
with tab_audio:
    from app.components.audio_player import render_audio_player
    uploaded_file.seek(0)
    render_audio_player(
        audio_bytes=uploaded_file.read(),
        waveform=waveform,
        sample_rate=SAMPLE_RATE,
        filename=uploaded_file.name,
    )

# ── Tab 2: Spectrogram ────────────────────────────────────────────────────────
with tab_spec:
    from app.components.spectrogram_view import render_spectrogram
    spec = single_result.get("spec")
    if spec is not None:
        render_spectrogram(
            spec,
            sample_rate=SAMPLE_RATE,
            title=f"Log-Mel Spectrogram — {uploaded_file.name}",
        )

# ── Tab 3: Prediction Result ──────────────────────────────────────────────────
with tab_result:
    from app.components.prediction_panel import render_prediction_panel
    render_prediction_panel(result)

# ── Tab 4: Window Analysis ────────────────────────────────────────────────────
with tab_windows:
    from app.components.window_analysis import render_window_analysis
    if use_windowed and "windows" in result:
        render_window_analysis(result["windows"], threshold=threshold)
    else:
        st.info(
            "Window analysis is disabled or not applicable for this file. "
            "Enable 'Window-based analysis' in the sidebar and upload again."
        )

# ── Tab 5: Grad-CAM ───────────────────────────────────────────────────────────
with tab_gradcam:
    from app.components.gradcam_view import render_gradcam
    if show_gradcam:
        gradcam_result = single_result.get("gradcam")
        spec           = single_result.get("spec")
        if spec is not None:
            render_gradcam(gradcam_result, spec)
        else:
            st.warning("Spectrogram not available for Grad-CAM.")
    else:
        st.info("Enable 'Show Grad-CAM explanation' in the sidebar.")

# ── Tab 6: Downloadable JSON Report ──────────────────────────────────────────
with tab_report:
    st.markdown("### 📥 Download Analysis Report")
    st.markdown("Download the complete analysis as a structured JSON report.")

    # Build clean report (no large arrays)
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

    st.download_button(
        label="⬇️ Download JSON Report",
        data=report_json,
        file_name=f"deepshield_report_{Path(uploaded_file.name).stem}.json",
        mime="application/json",
    )

    st.markdown("**Preview:**")
    with st.expander("📄 Report JSON", expanded=True):
        st.json(report)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    f"""
    <div style="text-align: center; color: #444; font-size: 0.8rem; padding: 10px 0;">
        🛡️ <strong style="color: #4ECDC4;">DeepShield Audio</strong> v1.0
        &nbsp;|&nbsp; Model: <strong>{selected_model_name}</strong>
        &nbsp;|&nbsp; Total inference: <strong>{total_ms:.0f} ms</strong>
        &nbsp;|&nbsp; Trained on ASVspoof 2019 LA dataset
        &nbsp;|&nbsp;
        <a href="https://github.com/yourusername/deepshield-audio" style="color: #4ECDC4;">
            GitHub ↗
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
