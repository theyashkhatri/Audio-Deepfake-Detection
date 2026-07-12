# DeepShield Audio — Task Tracker

## Phase 1 — Project Scaffold & Configuration
- [x] Create project directory structure
- [x] `src/config.py` — Central configuration
- [x] `requirements.txt`
- [x] `setup.py`
- [x] `README.md`
- [x] `.gitignore`

## Phase 2 — Data Layer
- [x] `src/__init__.py`
- [x] `src/data_parser.py` — Protocol-based dataset parser
- [x] `src/preprocessor.py` — Audio → Log-Mel (shared train/infer)
- [x] `src/dataset.py` — tf.data pipeline builder
- [x] `src/utils.py` — Plotting & logging helpers

## Phase 3 — Models
- [x] `src/models/__init__.py`
- [x] `src/models/custom_cnn.py`
- [x] `src/models/cnn_lstm.py`
- [x] `src/models/efficientnet.py`

## Phase 4 — Training Infrastructure
- [x] `src/trainer.py`

## Phase 5 — Evaluation
- [x] `src/evaluator.py`

## Phase 6 — Explainability
- [x] `src/explainability.py`

## Phase 7 — Threshold & Inference
- [x] `src/threshold_optimizer.py`
- [x] `src/inference.py`

## Phase 8 — Jupyter Notebooks
- [x] `notebooks/01_EDA.ipynb`
- [x] `notebooks/02_Preprocessing.ipynb`
- [x] `notebooks/03_Model_Training.ipynb`
- [x] `notebooks/04_Evaluation.ipynb`
- [x] `notebooks/05_Explainability.ipynb`

## Phase 9 — Streamlit Application
- [x] `app/streamlit_app.py`
- [x] `app/components/__init__.py`
- [x] `app/components/audio_player.py`
- [x] `app/components/spectrogram_view.py`
- [x] `app/components/prediction_panel.py`
- [x] `app/components/window_analysis.py`
- [x] `app/components/gradcam_view.py`

## Phase 10 — Unit Tests
- [x] `tests/__init__.py`
- [x] `tests/test_preprocessor.py`
- [x] `tests/test_data_parser.py`
- [x] `tests/test_inference.py`
- [x] `tests/test_models.py`
