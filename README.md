# 🛡️ DeepShield Audio — AI-Generated Voice & Audio Deepfake Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-ASVspoof%202019%20LA-purple)

**A complete end-to-end deep learning pipeline for detecting AI-generated and spoofed speech.**

[📱 Demo App](#streamlit-application) · [📋 Dataset](#dataset-setup) · [🧠 Models](#models) · [📊 Evaluation](#evaluation) · [🔥 Grad-CAM](#explainability)

</div>

---

## 📖 Overview

**DeepShield Audio** classifies speech audio as **Real (Bonafide)** or **Fake (Spoof/AI-Generated)** using three deep learning architectures trained on the [ASVspoof 2019 Logical Access (LA)](https://datashare.ed.ac.uk/handle/10283/3336) dataset.

### Key Features
- **3 Deep Learning Models**: Custom CNN, CNN-LSTM Hybrid, EfficientNetB0 Transfer Learning
- **7 Evaluation Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, EER, Confusion Matrix
- **Grad-CAM Explainability**: Visual explanation of model decisions on spectrograms
- **Decision Threshold Optimisation**: EER-based and F1-based threshold tuning
- **Long Audio Support**: Sliding-window inference for any-length audio
- **Professional Streamlit App**: WAV/MP3/FLAC upload, real-time analysis, JSON report
- **Zero Data Leakage**: Official ASVspoof protocol files strictly separate train/dev/eval
- **Identical Preprocessing**: Single `preprocessor.py` used at both train and inference time

---

## 🗂️ Project Structure

```
deepshield-audio/
├── data/
│   └── ASVspoof2019_LA/              # ← Place dataset here (NOT committed)
│       ├── ASVspoof2019_LA_train/flac/
│       ├── ASVspoof2019_LA_dev/flac/
│       ├── ASVspoof2019_LA_eval/flac/
│       └── ASVspoof2019_LA_cm_protocols/
│           ├── ASVspoof2019.LA.cm.train.trn.txt
│           ├── ASVspoof2019.LA.cm.dev.trl.txt
│           └── ASVspoof2019.LA.cm.eval.trl.txt
│
├── src/                              # Core library
│   ├── config.py                     # Central config (SR, n_mels, paths)
│   ├── data_parser.py                # Protocol-based parser (no leakage)
│   ├── preprocessor.py               # Audio → Log-Mel (shared train/infer)
│   ├── dataset.py                    # tf.data pipeline builder
│   ├── trainer.py                    # Training loop + callbacks
│   ├── evaluator.py                  # Metrics: Acc, P, R, F1, AUC, EER, CM
│   ├── explainability.py             # Grad-CAM implementation
│   ├── threshold_optimizer.py        # Decision threshold tuning
│   ├── inference.py                  # Single + windowed inference engine
│   ├── utils.py                      # Plotting, logging, JSON helpers
│   └── models/
│       ├── custom_cnn.py             # 4-block CNN architecture
│       ├── cnn_lstm.py               # CNN-LSTM hybrid
│       └── efficientnet.py           # EfficientNetB0 transfer learning
│
├── app/                              # Streamlit application
│   ├── streamlit_app.py              # Main app (6-tab UI)
│   └── components/
│       ├── audio_player.py           # Waveform player
│       ├── spectrogram_view.py       # Log-Mel spectrogram viewer
│       ├── prediction_panel.py       # Verdict, risk, confidence
│       ├── window_analysis.py        # Per-window timeline
│       └── gradcam_view.py           # Grad-CAM overlay
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Dataset EDA
│   ├── 02_Preprocessing.ipynb        # Feature extraction walkthrough
│   ├── 03_Model_Training.ipynb       # Train all three models
│   ├── 04_Evaluation.ipynb           # Compare models, select best
│   └── 05_Explainability.ipynb       # Grad-CAM analysis
│
├── tests/
│   ├── test_preprocessor.py          # 20+ preprocessing tests
│   ├── test_data_parser.py           # Protocol parsing tests
│   ├── test_inference.py             # Inference engine tests
│   └── test_models.py               # Model build + forward pass tests
│
├── saved_models/                     # Trained model checkpoints (.keras)
├── results/                          # Evaluation plots + metric JSONs
├── requirements.txt
├── setup.py
└── pytest.ini
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/deepshield-audio.git
cd deepshield-audio

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .                  # Editable install
```

### 2. Dataset Setup

Download the [ASVspoof 2019 LA dataset](https://datashare.ed.ac.uk/handle/10283/3336) from the University of Edinburgh DataShare:

```bash
# Create dataset directory
mkdir -p data/ASVspoof2019_LA

# After downloading, extract to:
# data/ASVspoof2019_LA/ASVspoof2019_LA_train/
# data/ASVspoof2019_LA/ASVspoof2019_LA_dev/
# data/ASVspoof2019_LA/ASVspoof2019_LA_eval/
# data/ASVspoof2019_LA/ASVspoof2019_LA_cm_protocols/
```

> **Dataset size**: ~10 GB. Contains 121,461 audio clips (train: 2,580 bonafide + 22,800 spoof; dev: 2,548 bonafide + 22,296 spoof; eval: 7,355 bonafide + 63,882 spoof).

### 3. Run Unit Tests (no dataset needed)

```bash
python -m pytest tests/ -v
```

### 4. Train Models

**Option A: Jupyter Notebook** (recommended)
```bash
jupyter notebook notebooks/03_Model_Training.ipynb
```

**Option B: Command Line**
```bash
# Train all three models
python -m src.trainer --model all --epochs 50

# Train single model
python -m src.trainer --model custom_cnn --epochs 30

# Quick pipeline test (5% of data)
python -m src.trainer --model custom_cnn --quick
```

### 5. Evaluate & Compare

```bash
# Evaluate on eval split
python -m src.evaluator --model all --split eval

# Or use the notebook:
jupyter notebook notebooks/04_Evaluation.ipynb
```

### 6. Launch Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 🧠 Models

### Custom CNN
```
Input (128, 251, 1) — Log-Mel spectrogram
→ Block 1: Conv2D(32×2) + BN + ReLU → MaxPool(2×2) → Dropout(0.2)
→ Block 2: Conv2D(64×2) + BN + ReLU → MaxPool(2×2) → Dropout(0.2)
→ Block 3: Conv2D(128×2)+ BN + ReLU → MaxPool(2×2) → Dropout(0.25)
→ Block 4: Conv2D(256)  + BN + ReLU           [← Grad-CAM target]
→ GlobalAveragePooling2D
→ Dense(256, ReLU) → Dropout(0.5)
→ Dense(1, sigmoid) → P(real)
```
**~3.2M parameters** | Fast training | CPU-friendly

### CNN-LSTM Hybrid
```
Input (128, 251, 1)
→ CNN blocks (Conv2D × 3) → Permute + Reshape → (time_steps, features)
→ Bidirectional LSTM(128) → LSTM(64) → Dropout(0.4)
→ Dense(128, ReLU) → Dense(1, sigmoid)
```
**~2.8M parameters** | Captures temporal rhythm/prosody patterns

### EfficientNetB0 Transfer Learning
```
Input (224, 224, 3) — resized + 3-channel spectrogram
→ EfficientNetB0 backbone (ImageNet pretrained)
   Phase 1: Backbone frozen → Train head
   Phase 2: Unfreeze block5a+ → Fine-tune at lr=1e-5
→ GlobalAveragePooling → Dense(256) → Dense(64) → Dense(1, sigmoid)
```
**~4.2M parameters** | Highest accuracy potential | Requires GPU for fast training

---

## 🎵 Audio Preprocessing

All preprocessing is performed by the **single shared `preprocessor.py`** — identical at train and inference time.

```
Audio File (WAV/FLAC/MP3)
        ↓
Load at 16,000 Hz (mono)
        ↓
Pad or Trim to 4.0 s (64,000 samples)
        ↓
Log-Mel Spectrogram
  n_mels=128, n_fft=1024, hop=256, win=1024
  fmin=0 Hz, fmax=8000 Hz, power=2.0
  → librosa.power_to_db(), top_db=80
        ↓
Shape: (128, 251)  [N_MELS × T_FRAMES]
        ↓
Global Standardisation (mean/std from train split)
        ↓
Add channel dim → (128, 251, 1)  [for CNN input]
```

---

## 📊 Evaluation

Models are evaluated on the official ASVspoof 2019 LA evaluation split with 7 metrics:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision | TP / (TP + FP) — fake precision |
| Recall | TP / (TP + FN) — fake recall |
| F1 Score | Harmonic mean of Precision & Recall |
| ROC-AUC | Area under ROC curve (↑ better) |
| **EER** | Equal Error Rate — FAR = FRR (↓ better) |
| Confusion Matrix | Full TP/FP/TN/FN breakdown |

> **Note**: Actual metrics depend on your trained model and training duration. Run the evaluation notebook for real results.

---

## 🛡️ Data Leakage Prevention

| Risk | Mitigation |
|------|-----------|
| Train/Dev/Eval overlap | Official ASVspoof2019 protocol `.txt` files — strictly disjoint |
| Preprocessing drift | Single shared `preprocessor.py` for all stages |
| Normalisation leak | Global stats computed **only on train split** |
| Model selection | Best model selected on **dev split**; eval used exactly **once** |
| Verified by code | `verify_no_leakage()` asserts zero file_id overlap |

---

## 🔥 Explainability (Grad-CAM)

Grad-CAM highlights which time-frequency regions of the Log-Mel spectrogram the model used for its decision:

```python
from src.explainability import get_gradcam_for_model
from src.preprocessor import preprocess_file

spec   = preprocess_file("audio.flac")
gradcam = get_gradcam_for_model(model, model_name="custom_cnn")
result  = gradcam.explain(spec)

# result["heatmap"]  — raw (H', W') heatmap
# result["overlay"]  — RGB blend of spectrogram + heatmap
```

---

## ⚡ Inference API

```python
from src.inference import AudioInferenceEngine
from src.trainer import load_model

model  = load_model("custom_cnn")
engine = AudioInferenceEngine(model, model_name="custom_cnn", threshold=0.5)

# Single file
result = engine.predict("audio.wav")
print(f"Label: {result['label']}")
print(f"P(fake): {result['fake_prob']:.4f}")
print(f"Risk: {result['risk']['level']}")

# Long audio (windowed)
windowed = engine.predict_long_audio("long_audio.wav", window_sec=3.0, hop_sec=1.5)
print(f"Windows: {windowed['n_windows']}")
for w in windowed["windows"]:
    print(f"  [{w['start_sec']:.1f}s-{w['end_sec']:.1f}s] P(fake)={w['fake_prob']:.3f}")
```

---

## 🖥️ Streamlit Application

```
┌─────────────────────────────────────────────────────┐
│  🛡️ DeepShield Audio — AI Deepfake Detector         │
├──────────┬──────────────────────────────────────────┤
│ Sidebar  │  📂 Upload Audio (WAV/MP3/FLAC)           │
│ - Model  ├──────────────────────────────────────────┤
│ - Thresh │  Tabs: Audio | Spectrogram | Result | ... │
│ - Window ├──────────────────────────────────────────┤
│ - GradCAM│  🎵 Waveform       │  🎨 Spectrogram     │
│          ├──────────────────────────────────────────┤
│          │  🎯 FAKE — AI GENERATED (94.2%)           │
│          │  🔴 Risk: HIGH                            │
│          │  P(Real): 5.8%  P(Fake): 94.2%           │
│          ├──────────────────────────────────────────┤
│          │  📊 Window Analysis Timeline              │
│          ├──────────────────────────────────────────┤
│          │  🔥 Grad-CAM Explanation                  │
│          ├──────────────────────────────────────────┤
│          │  📥 Download JSON Report                  │
└──────────┴──────────────────────────────────────────┘
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_preprocessor.py -v

# Exclude slow tests
python -m pytest tests/ -m "not slow"
```

Test coverage includes:
- ✅ `test_preprocessor.py` — 20+ tests for audio loading, pad/trim, Log-Mel, normalisation
- ✅ `test_data_parser.py`  — Protocol parsing, leakage detection, label encoding
- ✅ `test_models.py`       — Build, forward pass, Grad-CAM layer existence, gradients
- ✅ `test_inference.py`    — Single/windowed inference, risk bands, threshold effects

---

## 📓 Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_EDA.ipynb` | Dataset statistics, class distributions, spoofing algorithm breakdown |
| `02_Preprocessing.ipynb` | Feature extraction walkthrough, hyperparameter impact |
| `03_Model_Training.ipynb` | Train all three models, plot learning curves |
| `04_Evaluation.ipynb` | Compare models, ROC curves, select best by EER |
| `05_Explainability.ipynb` | Grad-CAM on bonafide vs. spoof samples |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | 2.15 | Deep learning framework |
| librosa | 0.10.1 | Audio processing + Log-Mel |
| streamlit | 1.35 | Web application |
| scikit-learn | 1.5 | Metrics + preprocessing |
| plotly | 5.22 | Interactive charts |
| soundfile | 0.12.1 | Audio file I/O |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Citation

If you use this work, please cite:

```bibtex
@dataset{asvspoof2019,
  title     = {ASVspoof 2019: A large-scale public database},
  author    = {Wang, Xin and others},
  year      = {2020},
  publisher = {University of Edinburgh},
  url       = {https://datashare.ed.ac.uk/handle/10283/3336}
}
```

---

<div align="center">
Built with ❤️ using TensorFlow, Librosa, and Streamlit
</div>
