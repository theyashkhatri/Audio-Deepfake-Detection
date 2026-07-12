"""
DeepShield Audio — Decision Threshold Optimiser
=================================================
Sweeps decision thresholds on the DEV split to find the optimal
operating point for the production deployment of each model.

Three optimisation objectives:
  1. EER threshold  — equates FAR and FRR (standard anti-spoofing)
  2. Max-F1 threshold — maximises harmonic mean of precision/recall
  3. Balanced accuracy threshold — maximises (TPR + TNR) / 2

Results are persisted to JSON for use in inference.py.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, balanced_accuracy_score

try:
    from src.config import RESULTS_DIR, DEFAULT_THRESHOLD
    from src.utils import save_json, load_json
except ImportError:
    from config import RESULTS_DIR, DEFAULT_THRESHOLD
    from utils import save_json, load_json

logger = logging.getLogger(__name__)

# Path where per-model optimal thresholds are persisted
THRESHOLD_STORE_PATH = RESULTS_DIR / "optimal_thresholds.json"


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def sweep_thresholds(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_steps: int = 200,
) -> Dict[str, float]:
    """
    Sweep decision thresholds from 0 to 1 and compute F1 and balanced accuracy.

    Args:
        y_true:    Binary ground truth (1=real, 0=spoof).
        y_scores:  Model output P(real) scores.
        n_steps:   Number of threshold points to evaluate.

    Returns:
        Dict with:
          'max_f1_threshold':           threshold maximising F1
          'max_f1_value':               achieved F1
          'balanced_acc_threshold':     threshold maximising balanced accuracy
          'balanced_acc_value':         achieved balanced accuracy
          'eer_threshold':              EER threshold
          'eer_value':                  EER value
    """
    thresholds = np.linspace(0.0, 1.0, n_steps + 1)

    f1_scores   = []
    bal_accs    = []

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        bal_accs.append(balanced_accuracy_score(y_true, y_pred))

    f1_scores = np.array(f1_scores)
    bal_accs  = np.array(bal_accs)

    best_f1_idx   = np.argmax(f1_scores)
    best_bacc_idx = np.argmax(bal_accs)

    # EER threshold
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr     = 1.0 - tpr
    abs_diff = np.abs(fpr - fnr)
    eer_idx  = np.argmin(abs_diff)
    eer_val  = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_thr  = float(roc_thresholds[eer_idx])

    return {
        "max_f1_threshold":       float(thresholds[best_f1_idx]),
        "max_f1_value":           float(f1_scores[best_f1_idx]),
        "balanced_acc_threshold": float(thresholds[best_bacc_idx]),
        "balanced_acc_value":     float(bal_accs[best_bacc_idx]),
        "eer_threshold":          eer_thr,
        "eer_value":              eer_val,
        "all_thresholds":         thresholds.tolist(),
        "all_f1_scores":          f1_scores.tolist(),
        "all_balanced_accs":      bal_accs.tolist(),
    }


def find_optimal_threshold(
    model,
    dev_dataset,
    model_name: str = "model",
    objective: str = "eer",        # 'eer' | 'max_f1' | 'balanced_acc'
    save_plots: bool = True,
    save_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Run the full threshold optimisation pipeline for a model on the dev split.

    Args:
        model:       Keras model (compiled, loaded).
        dev_dataset: tf.data.Dataset (dev split).
        model_name:  For logging and file naming.
        objective:   Which threshold to use as the primary recommendation.
        save_plots:  Whether to save the threshold sweep plot.
        save_dir:    Output directory.

    Returns:
        Dict with all threshold results + primary 'recommended_threshold'.
    """
    save_dir = Path(save_dir) if save_dir else RESULTS_DIR / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Optimising threshold for %s (objective=%s) …", model_name, objective)

    # Collect scores
    all_labels: List[int]   = []
    all_scores: List[float] = []

    for batch_specs, batch_labels in dev_dataset:
        scores = model(batch_specs, training=False).numpy().flatten()
        labels = batch_labels.numpy().flatten()
        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())

    y_true   = np.array(all_labels)
    y_scores = np.array(all_scores)

    # Sweep
    sweep_result = sweep_thresholds(y_true, y_scores)

    # Select recommended threshold
    objective_map = {
        "eer":          "eer_threshold",
        "max_f1":       "max_f1_threshold",
        "balanced_acc": "balanced_acc_threshold",
    }
    recommended_key = objective_map.get(objective, "eer_threshold")
    recommended = sweep_result[recommended_key]

    result = {
        "model_name":            model_name,
        "objective":             objective,
        "recommended_threshold": recommended,
        **sweep_result,
    }

    # Remove large arrays from JSON (keep only summary)
    result_to_save = {k: v for k, v in result.items()
                      if "all_" not in k}
    save_json(result_to_save, save_dir / f"{model_name}_threshold_opt.json")

    logger.info(
        "%s optimal thresholds:\n"
        "  EER:           %.4f (thr=%.4f)\n"
        "  Max-F1:        %.4f (thr=%.4f)\n"
        "  Balanced Acc:  %.4f (thr=%.4f)\n"
        "  → Recommended: %.4f (%s)",
        model_name,
        sweep_result["eer_value"],           sweep_result["eer_threshold"],
        sweep_result["max_f1_value"],        sweep_result["max_f1_threshold"],
        sweep_result["balanced_acc_value"],  sweep_result["balanced_acc_threshold"],
        recommended, objective,
    )

    # Plot
    if save_plots:
        _plot_threshold_sweep(sweep_result, model_name, recommended,
                              save_path=save_dir / f"{model_name}_threshold_sweep.png")

    return result


def _plot_threshold_sweep(
    sweep: dict,
    model_name: str,
    recommended: float,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot F1 and balanced accuracy vs threshold."""
    thresholds = np.array(sweep["all_thresholds"])
    f1_scores  = np.array(sweep["all_f1_scores"])
    bal_accs   = np.array(sweep["all_balanced_accs"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, f1_scores, "b-",  linewidth=2, label="F1 Score")
    ax.plot(thresholds, bal_accs,  "g--", linewidth=2, label="Balanced Accuracy")
    ax.axvline(recommended, color="red", linewidth=2.5, linestyle=":",
               label=f"Recommended threshold = {recommended:.3f}")
    ax.axvline(sweep["eer_threshold"], color="orange", linewidth=1.5, linestyle="-.",
               label=f"EER threshold = {sweep['eer_threshold']:.3f}")

    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"{model_name} — Threshold Optimisation", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_optimal_thresholds(thresholds_per_model: Dict[str, float]) -> None:
    """
    Persist optimal thresholds for all models to a shared JSON file.

    Args:
        thresholds_per_model: {model_name: recommended_threshold}
    """
    existing = {}
    if THRESHOLD_STORE_PATH.exists():
        existing = load_json(THRESHOLD_STORE_PATH)
    existing.update(thresholds_per_model)
    save_json(existing, THRESHOLD_STORE_PATH)
    logger.info("Thresholds saved → %s", THRESHOLD_STORE_PATH)


def load_optimal_threshold(model_name: str, fallback: float = DEFAULT_THRESHOLD) -> float:
    """
    Load the optimal threshold for a model from the persisted store.

    Args:
        model_name: Model name key.
        fallback:   Default threshold if none found.

    Returns:
        Float threshold in [0, 1].
    """
    if THRESHOLD_STORE_PATH.exists():
        store = load_json(THRESHOLD_STORE_PATH)
        if model_name in store:
            return float(store[model_name])
    logger.warning("No saved threshold for %s — using fallback %.2f", model_name, fallback)
    return fallback
