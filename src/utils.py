"""
DeepShield Audio — Utility Functions
=====================================
Plotting helpers, logging setup, JSON I/O, and visualisation utilities.
"""

import json
import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

try:
    from src.config import RESULTS_DIR
except ImportError:
    from config import RESULTS_DIR

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
) -> None:
    """Configure root logger with console + optional file handler."""
    handlers: Dict = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "detailed",
        }
    }
    handler_names = ["console"]

    if log_file:
        handlers["file"] = {
            "class": "logging.FileHandler",
            "filename": str(log_file),
            "formatter": "detailed",
        }
        handler_names.append("file")

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": handlers,
        "root": {"level": level, "handlers": handler_names},
    })


# ─────────────────────────────────────────────────────────────────────────────
# JSON I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_json(data: dict, path: Union[str, Path]) -> None:
    """Save dictionary to JSON file (handles numpy types)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
            return super().default(obj)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)
    logging.getLogger(__name__).info("Saved JSON → %s", path)


def load_json(path: Union[str, Path]) -> dict:
    """Load JSON file into a dictionary."""
    with open(path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# SPECTROGRAM PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_spectrogram(
    spec: np.ndarray,
    title: str = "Log-Mel Spectrogram",
    sr: int = 16_000,
    hop_length: int = 256,
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "magma",
) -> plt.Figure:
    """
    Visualise a (N_MELS, T_FRAMES) Log-Mel spectrogram.

    Args:
        spec: 2-D array. If 3-D (H, W, 1) the trailing dim is squeezed.
        save_path: If provided, saves the figure.
        ax: Existing axes to draw on (no new figure created).
    """
    import librosa.display

    if spec.ndim == 3:
        spec = spec.squeeze(-1)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    librosa.display.specshow(
        spec, sr=sr, hop_length=hop_length,
        x_axis="time", y_axis="mel",
        ax=ax, cmap=cmap,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel Frequency")

    if standalone:
        plt.colorbar(ax.images[0], ax=ax, format="%+2.0f dB")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
    return fig


def plot_waveform(
    waveform: np.ndarray,
    sr: int = 16_000,
    title: str = "Waveform",
    save_path: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot audio waveform as a time-domain amplitude plot."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 2))
    else:
        fig = ax.get_figure()

    time_axis = np.linspace(0, len(waveform) / sr, num=len(waveform))
    ax.plot(time_axis, waveform, linewidth=0.4, color="#4ECDC4")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim([0, time_axis[-1]])
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    if standalone:
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(
    history: dict,
    model_name: str = "Model",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot training & validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["loss"],     "b-o", markersize=3, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", markersize=3, label="Val Loss")
    axes[0].set_title(f"{model_name} — Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    acc_key = "binary_accuracy" if "binary_accuracy" in history else "accuracy"
    val_acc_key = f"val_{acc_key}"
    if acc_key in history:
        axes[1].plot(epochs, history[acc_key],     "b-o", markersize=3, label="Train Acc")
        axes[1].plot(epochs, history[val_acc_key], "r-o", markersize=3, label="Val Acc")
        axes[1].set_title(f"{model_name} — Accuracy", fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])

    plt.suptitle(f"{model_name} Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ["Spoof (Fake)", "Bonafide (Real)"],
    model_name: str = "Model",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot a styled confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="gray",
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )
    ax.set_title(f"{model_name}\nConfusion Matrix", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ROC CURVE
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    results: Dict[str, dict],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Overlay ROC curves for multiple models.

    Args:
        results: Dict {model_name: {"fpr": array, "tpr": array, "roc_auc": float}}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]

    for (model_name, res), color in zip(results.items(), colors):
        fpr = res.get("fpr", [])
        tpr = res.get("tpr", [])
        auc = res.get("roc_auc", 0.0)
        if len(fpr) and len(tpr):
            ax.plot(fpr, tpr, color=color, linewidth=2.5,
                    label=f"{model_name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate (FAR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (1 - FRR)", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_table(
    metrics_df,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Render a metrics comparison table as a matplotlib figure."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")

    col_labels = list(metrics_df.columns)
    row_labels  = list(metrics_df.index)
    cell_vals   = metrics_df.values.tolist()

    table = ax.table(
        cellText=cell_vals,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Colour header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#264653")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    ax.set_title("Model Comparison — All Metrics", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW ANALYSIS PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_window_analysis(
    window_results: List[dict],
    title: str = "Window-Based Fake Probability Analysis",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Bar chart of per-window fake probability over time.

    Args:
        window_results: List of dicts with keys:
            'window_idx', 'start_sec', 'end_sec', 'fake_prob', 'label'
    """
    if not window_results:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No windows to display", ha="center", va="center")
        return fig

    starts    = [w["start_sec"] for w in window_results]
    fake_probs = [w["fake_prob"] for w in window_results]
    widths    = [w["end_sec"] - w["start_sec"] for w in window_results]

    colors = ["#E63946" if p > 0.5 else "#2A9D8F" for p in fake_probs]

    fig, ax = plt.subplots(figsize=(max(10, len(window_results) * 1.2), 4))
    bars = ax.bar(starts, fake_probs, width=[w * 0.8 for w in widths],
                  align="edge", color=colors, edgecolor="white", linewidth=0.5)

    ax.axhline(0.5, color="black", linewidth=1.5, linestyle="--", label="Decision threshold (0.5)")
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("P(Fake)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Colour legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#E63946", label="Fake window"),
        Patch(facecolor="#2A9D8F", label="Real window"),
    ]
    ax.legend(handles=legend_patches + ax.get_legend_handles_labels()[0][:1],
              loc="upper right", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
