"""
DeepShield Audio — Training Module
=====================================
Provides a reusable training loop with callbacks for all three model
architectures. Saves the best checkpoint based on val_loss.

Usage:
    from src.trainer import train_model
    history = train_model(model, train_ds, dev_ds, model_name="custom_cnn")
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

import tensorflow as tf
try:
    # Keras 3 (TF 2.16+)
    import keras
except ImportError:
    # Keras 2 (TF 2.15)
    from tensorflow import keras

try:
    from src.config import (
        EPOCHS, BATCH_SIZE, SAVED_MODELS_DIR, LOGS_DIR,
        MODEL_CUSTOM_CNN, MODEL_CNN_LSTM, MODEL_EFFICIENTNET,
    )
    from src.utils import plot_training_history, save_json
except ImportError:
    from config import (
        EPOCHS, BATCH_SIZE, SAVED_MODELS_DIR, LOGS_DIR,
        MODEL_CUSTOM_CNN, MODEL_CNN_LSTM, MODEL_EFFICIENTNET,
    )
    from utils import plot_training_history, save_json

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def build_callbacks(
    model_name: str,
    checkpoint_dir: Path,
    tensorboard_dir: Path,
    patience_early: int = 10,
    patience_lr: int = 5,
    min_lr: float = 1e-7,
    lr_factor: float = 0.5,
) -> list:
    """
    Build a standard set of Keras callbacks.

    Returns:
        List of Keras callbacks:
          - ModelCheckpoint (saves best val_loss)
          - EarlyStopping
          - ReduceLROnPlateau
          - TensorBoard
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}_best.keras"

    callbacks = [
        # Save the best model weights
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        # Stop training when val_loss stops improving
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_early,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce LR when stuck
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=lr_factor,
            patience=patience_lr,
            min_lr=min_lr,
            verbose=1,
        ),
        # TensorBoard logs
        keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir / model_name),
            histogram_freq=0,
            update_freq="epoch",
        ),
        # CSV logger
        keras.callbacks.CSVLogger(
            str(checkpoint_dir / f"{model_name}_training_log.csv"),
            append=False,
        ),
    ]
    return callbacks


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    model_name: str = "model",
    epochs: int = EPOCHS,
    class_weights: Optional[Dict[int, float]] = None,
    save_dir: Optional[Path] = None,
    tensorboard_dir: Optional[Path] = None,
    patience_early: int = 10,
    patience_lr: int = 5,
    initial_epoch: int = 0,
) -> Dict[str, Any]:
    """
    Train a compiled Keras model with standard callbacks.

    Args:
        model:          Compiled Keras model.
        train_ds:       Training tf.data.Dataset.
        val_ds:         Validation tf.data.Dataset.
        model_name:     Used for checkpoint naming and directory structure.
        epochs:         Maximum training epochs.
        class_weights:  Optional dict {0: w0, 1: w1} for imbalanced data.
        save_dir:       Where to save model checkpoints (default: saved_models/).
        tensorboard_dir: Where to write TensorBoard logs.
        patience_early: EarlyStopping patience.
        patience_lr:    ReduceLROnPlateau patience.
        initial_epoch:  For resuming training.

    Returns:
        Dict with:
          'history':       Keras History.history dict (loss, val_loss, …)
          'best_model_path': Path to saved best model (.keras file)
          'training_time_sec': Wall-clock training time
          'model_name':    model_name string
    """
    save_dir       = Path(save_dir)       if save_dir       else SAVED_MODELS_DIR / model_name
    tensorboard_dir = Path(tensorboard_dir) if tensorboard_dir else LOGS_DIR / "tensorboard"

    save_dir.mkdir(parents=True, exist_ok=True)

    callbacks = build_callbacks(
        model_name=model_name,
        checkpoint_dir=save_dir,
        tensorboard_dir=tensorboard_dir,
        patience_early=patience_early,
        patience_lr=patience_lr,
    )

    logger.info("=" * 60)
    logger.info("Training: %s | Epochs: %d | Class weights: %s",
                model_name, epochs, class_weights)
    logger.info("=" * 60)

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - start_time

    logger.info("Training complete in %.1f s (%.1f min)", elapsed, elapsed / 60)

    # Save training history
    history_dict = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    save_json(history_dict, save_dir / f"{model_name}_history.json")

    # Plot curves
    from src.utils import plot_training_history
    fig = plot_training_history(
        history.history,
        model_name=model_name,
        save_path=save_dir / f"{model_name}_training_curves.png",
    )
    import matplotlib.pyplot as plt
    plt.close(fig)

    best_path = save_dir / f"{model_name}_best.keras"
    logger.info("Best model saved → %s", best_path)

    return {
        "history":            history_dict,
        "best_model_path":    str(best_path),
        "training_time_sec":  elapsed,
        "model_name":         model_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FINE-TUNING HELPER (EfficientNet Phase 2)
# ─────────────────────────────────────────────────────────────────────────────

def finetune_efficientnet(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    finetune_epochs: int = 20,
    learning_rate: float = 1e-5,
    save_dir: Optional[Path] = None,
    class_weights: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """
    Phase 2: Fine-tune EfficientNetB0 with unfrozen top layers.

    Args:
        model:            Phase 1 trained model.
        finetune_epochs:  Number of additional fine-tuning epochs.
        learning_rate:    Reduced LR for fine-tuning.

    Returns:
        Same format as train_model().
    """
    try:
        from src.models.efficientnet import unfreeze_for_finetuning
    except ImportError:
        from models.efficientnet import unfreeze_for_finetuning

    model_name = "efficientnet_b0_finetuned"
    save_dir   = Path(save_dir) if save_dir else SAVED_MODELS_DIR / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    model = unfreeze_for_finetuning(model, learning_rate=learning_rate)
    logger.info("EfficientNet Phase 2 fine-tuning started | LR=%.2e", learning_rate)

    result = train_model(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        model_name=model_name,
        epochs=finetune_epochs,
        class_weights=class_weights,
        save_dir=save_dir,
        patience_early=7,
        patience_lr=3,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str, model_dir: Optional[Path] = None) -> keras.Model:
    """
    Load a saved Keras model from disk.

    Args:
        model_name: One of 'custom_cnn', 'cnn_lstm', 'efficientnet_b0'.
        model_dir:  Override default save directory.

    Returns:
        Loaded Keras model (compiled).
    """
    model_dir  = Path(model_dir) if model_dir else SAVED_MODELS_DIR / model_name
    model_path = model_dir / f"{model_name}_best.keras"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Have you trained the model yet? Run the training notebook first."
        )

    model = keras.models.load_model(str(model_path))
    logger.info("Loaded model from %s", model_path)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point: trains all three models sequentially."""
    import argparse

    parser = argparse.ArgumentParser(description="DeepShield Audio — Model Trainer")
    parser.add_argument("--model", choices=["custom_cnn", "cnn_lstm", "efficientnet", "all"],
                        default="all", help="Which model to train")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--quick", action="store_true",
                        help="Use 5%% of data for fast pipeline check")
    args = parser.parse_args()

    from src.utils import setup_logging
    setup_logging("INFO")

    logger.info("DeepShield Audio — Training started")
    logger.info("Model: %s | Epochs: %d | Quick: %s", args.model, args.epochs, args.quick)

    # Data loading
    from src.data_parser import load_all_splits, get_file_label_pairs
    from src.dataset import build_train_dev_datasets, get_class_weights

    splits = load_all_splits()
    train_pairs = get_file_label_pairs(splits["train"])
    dev_pairs   = get_file_label_pairs(splits["dev"])

    if args.quick:
        import random
        random.seed(42)
        k = max(10, len(train_pairs) // 20)
        train_pairs = random.sample(train_pairs, k)
        dev_pairs   = random.sample(dev_pairs, min(k // 5, len(dev_pairs)))
        logger.info("Quick mode: %d train, %d dev", len(train_pairs), len(dev_pairs))

    labels = [lbl for _, lbl in train_pairs]
    cw     = get_class_weights(labels)

    models_to_train = [args.model] if args.model != "all" else ["custom_cnn", "cnn_lstm", "efficientnet"]

    for model_type in models_to_train:
        if model_type == "custom_cnn":
            from src.models.custom_cnn import build_custom_cnn
            model = build_custom_cnn()
            train_ds, dev_ds = build_train_dev_datasets(train_pairs, dev_pairs,
                                                         model_type="cnn")
        elif model_type == "cnn_lstm":
            from src.models.cnn_lstm import build_cnn_lstm
            model = build_cnn_lstm()
            train_ds, dev_ds = build_train_dev_datasets(train_pairs, dev_pairs,
                                                         model_type="cnn")
        elif model_type == "efficientnet":
            from src.models.efficientnet import build_efficientnet_b0
            model = build_efficientnet_b0(freeze_backbone=True)
            train_ds, dev_ds = build_train_dev_datasets(train_pairs, dev_pairs,
                                                         model_type="efficientnet")

        result = train_model(model, train_ds, dev_ds,
                             model_name=model_type,
                             epochs=args.epochs,
                             class_weights=cw)
        logger.info("Done: %s → %s", model_type, result["best_model_path"])


if __name__ == "__main__":
    main()
