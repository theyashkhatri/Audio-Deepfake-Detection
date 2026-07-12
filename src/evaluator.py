"""
DeepShield Audio — Evaluator
==============================
Computes and reports all evaluation metrics:
  - Accuracy, Precision, Recall, F1 Score
  - ROC-AUC
  - Equal Error Rate (EER)  ← key anti-spoofing metric
  - Confusion Matrix

Designed to run on the EVAL split (used exactly ONCE for final numbers).
Model selection is done on the DEV split to prevent eval set leakage.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)

try:
    from src.config import RESULTS_DIR, DEFAULT_THRESHOLD
    from src.utils import (
        plot_confusion_matrix, plot_roc_curves, plot_metrics_table, save_json,
    )
except ImportError:
    from config import RESULTS_DIR, DEFAULT_THRESHOLD
    from utils import (
        plot_confusion_matrix, plot_roc_curves, plot_metrics_table, save_json,
    )

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# EER CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and the corresponding threshold.

    EER is the point on the DET curve where:
      False Acceptance Rate (FAR) == False Rejection Rate (FRR)

    In our convention:
      - Positive (1) = bonafide (real)
      - Negative (0) = spoof (fake)
      - FAR = FP / (FP + TN)  — spoofs accepted as real
      - FRR = FN / (FN + TP)  — real rejected as spoof

    Args:
        y_true:   Binary ground truth (1=real, 0=spoof).
        y_scores: Model output scores (probability of being real).

    Returns:
        (eer, threshold) — EER as fraction [0, 1], optimal threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr  # FRR

    # Find where FAR ≈ FRR
    abs_diff = np.abs(fpr - fnr)
    min_idx  = np.argmin(abs_diff)
    eer       = float((fpr[min_idx] + fnr[min_idx]) / 2.0)
    threshold = float(thresholds[min_idx])

    return eer, threshold


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    dataset,
    model_name: str = "Model",
    threshold: float = DEFAULT_THRESHOLD,
    save_dir: Optional[Path] = None,
) -> Dict:
    """
    Run full evaluation on a tf.data.Dataset and compute all metrics.

    Args:
        model:      Compiled / loaded Keras model.
        dataset:    tf.data.Dataset yielding (spec_batch, label_batch).
        model_name: For logging and plot titles.
        threshold:  Decision threshold for binary classification.
        save_dir:   Save plots and JSON here. Defaults to RESULTS_DIR.

    Returns:
        Dict with all metrics + raw arrays.
    """
    import tensorflow as tf

    save_dir = Path(save_dir) if save_dir else RESULTS_DIR / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Evaluating %s …", model_name)

    # Collect predictions
    all_labels: List[int]   = []
    all_scores: List[float] = []

    for batch_specs, batch_labels in dataset:
        scores = model(batch_specs, training=False)
        scores = scores.numpy().flatten()
        labels = batch_labels.numpy().flatten()
        all_scores.extend(scores.tolist())
        all_labels.extend(labels.tolist())

    y_true   = np.array(all_labels, dtype=int)
    y_scores = np.array(all_scores, dtype=float)   # P(real)
    y_pred   = (y_scores >= threshold).astype(int)

    # ── Core Metrics ──────────────────────────────────────────────────────────
    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        roc_auc = float("nan")

    eer, eer_threshold = compute_eer(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)

    # ── ROC Curve Arrays (for plotting) ──────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)

    # ── Log Summary ──────────────────────────────────────────────────────────
    logger.info(
        "\n%s Results:\n"
        "  Accuracy : %.4f\n"
        "  Precision: %.4f\n"
        "  Recall   : %.4f\n"
        "  F1 Score : %.4f\n"
        "  ROC-AUC  : %.4f\n"
        "  EER      : %.4f (threshold = %.4f)\n"
        "  Threshold: %.2f\n",
        model_name, acc, prec, rec, f1, roc_auc, eer, eer_threshold, threshold,
    )
    print(classification_report(y_true, y_pred,
                                  target_names=["Spoof", "Bonafide"],
                                  zero_division=0))

    # ── Plots ─────────────────────────────────────────────────────────────────
    cm_fig = plot_confusion_matrix(
        cm, model_name=model_name,
        save_path=save_dir / f"{model_name}_confusion_matrix.png",
    )
    import matplotlib.pyplot as plt
    plt.close(cm_fig)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        "model_name":    model_name,
        "threshold":     threshold,
        "accuracy":      acc,
        "precision":     prec,
        "recall":        rec,
        "f1_score":      f1,
        "roc_auc":       roc_auc,
        "eer":           eer,
        "eer_threshold": eer_threshold,
        "confusion_matrix": cm.tolist(),
        "fpr":           fpr.tolist(),
        "tpr":           tpr.tolist(),
        "n_samples":     len(y_true),
        "n_positive":    int(y_true.sum()),
        "n_negative":    int((1 - y_true).sum()),
    }
    save_json(result, save_dir / f"{model_name}_metrics.json")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(
    results: Dict[str, dict],
    save_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Create a side-by-side comparison DataFrame for multiple models.

    Args:
        results: Dict {model_name: metrics_dict} (output of evaluate_model).
        save_dir: Where to save plots and CSVs.

    Returns:
        pandas DataFrame with one row per model.
    """
    save_dir = Path(save_dir) if save_dir else RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, res in results.items():
        rows.append({
            "Model":     name,
            "Accuracy":  f"{res['accuracy']:.4f}",
            "Precision": f"{res['precision']:.4f}",
            "Recall":    f"{res['recall']:.4f}",
            "F1 Score":  f"{res['f1_score']:.4f}",
            "ROC-AUC":   f"{res['roc_auc']:.4f}",
            "EER (↓)":   f"{res['eer']:.4f}",
        })

    df = pd.DataFrame(rows).set_index("Model")
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70)

    # Save CSV
    df.to_csv(save_dir / "model_comparison.csv")
    logger.info("Comparison table saved → %s", save_dir / "model_comparison.csv")

    # Save ROC overlay
    roc_fig = plot_roc_curves(results, save_path=save_dir / "roc_curves.png")
    import matplotlib.pyplot as plt
    plt.close(roc_fig)

    # Save table image
    table_fig = plot_metrics_table(df, save_path=save_dir / "metrics_table.png")
    plt.close(table_fig)

    return df


def select_best_model(results: Dict[str, dict], metric: str = "eer") -> str:
    """
    Select the best model by a given metric.

    Args:
        results: Dict {model_name: metrics_dict}.
        metric:  'eer' (lower is better) or 'f1_score'/'roc_auc' (higher better).

    Returns:
        Name of the best model.
    """
    lower_is_better = metric in ("eer",)
    best_name = None
    best_val  = float("inf") if lower_is_better else float("-inf")

    for name, res in results.items():
        val = res.get(metric, float("nan"))
        if np.isnan(val):
            continue
        if (lower_is_better and val < best_val) or (not lower_is_better and val > best_val):
            best_val  = val
            best_name = name

    direction = "↓" if lower_is_better else "↑"
    logger.info("Best model by %s %s: %s = %.4f", metric, direction, best_name, best_val)
    return best_name


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Evaluate all trained models on the eval split."""
    import argparse

    parser = argparse.ArgumentParser(description="DeepShield Audio — Evaluator")
    parser.add_argument("--model", default="all",
                        choices=["custom_cnn", "cnn_lstm", "efficientnet_b0", "all"])
    parser.add_argument("--split", default="eval", choices=["dev", "eval"])
    args = parser.parse_args()

    from src.utils import setup_logging
    setup_logging("INFO")

    from src.data_parser import load_split, get_file_label_pairs
    from src.dataset import build_dataset
    from src.trainer import load_model
    from src.config import ALL_MODELS

    split_df = load_split(args.split)
    pairs    = get_file_label_pairs(split_df)

    models_to_eval = ALL_MODELS if args.model == "all" else [args.model]
    all_results = {}

    for model_name in models_to_eval:
        try:
            model = load_model(model_name)
            ds    = build_dataset(pairs, shuffle=False,
                                   model_type="efficientnet" if "efficient" in model_name else "cnn")
            result = evaluate_model(model, ds, model_name=model_name)
            all_results[model_name] = result
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", model_name, e)

    if len(all_results) > 1:
        df = compare_models(all_results)
        best = select_best_model(all_results, metric="eer")
        logger.info("🏆 Best model: %s", best)
    elif all_results:
        logger.info("Single model evaluated: %s", list(all_results.keys())[0])


if __name__ == "__main__":
    main()
