"""
DeepShield Audio — tf.data Pipeline Builder
=============================================
Constructs high-performance TensorFlow Dataset objects from parsed
file lists. All preprocessing is applied inside tf.data.map() using
tf.py_function wrappers for librosa compatibility.

Key design choices:
  - Shuffle only the training set (reproducibly with fixed seed)
  - Cache after preprocessing for fast epoch iteration
  - Prefetch to overlap CPU preprocessing with GPU compute
  - Identical preprocessing path as inference (via shared preprocessor)
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import tensorflow as tf

try:
    from src.preprocessor import preprocess_file
    from src.config import (
        N_MELS, T_FRAMES, BATCH_SIZE, SEED,
        EFFICIENTNET_INPUT_SIZE,
    )
except ImportError:
    from preprocessor import preprocess_file
    from config import (
        N_MELS, T_FRAMES, BATCH_SIZE, SEED,
        EFFICIENTNET_INPUT_SIZE,
    )

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING WRAPPER  (tf.py_function compatible)
# ─────────────────────────────────────────────────────────────────────────────

def _load_and_preprocess(path_tensor: tf.Tensor, label_tensor: tf.Tensor):
    """
    Wraps preprocess_file() for use inside tf.data.map().
    Called via tf.py_function so it can use numpy/librosa code.
    """
    path = path_tensor.numpy().decode("utf-8")
    try:
        spec = preprocess_file(path, add_channel_dim=True)   # (N_MELS, T_FRAMES, 1)
    except Exception as e:
        logger.error("Error preprocessing %s: %s", path, e)
        spec = np.zeros((N_MELS, T_FRAMES, 1), dtype=np.float32)
    return spec.astype(np.float32), label_tensor


def _tf_preprocess(path: tf.Tensor, label: tf.Tensor):
    """
    tf.py_function wrapper returning typed tensors.
    """
    spec, lbl = tf.py_function(
        func=_load_and_preprocess,
        inp=[path, label],
        Tout=[tf.float32, tf.int32],
    )
    spec.set_shape([N_MELS, T_FRAMES, 1])
    lbl.set_shape([])
    return spec, lbl


# ─────────────────────────────────────────────────────────────────────────────
# EfficientNet preprocessing (resizes spectrogram to 224×224×3)
# ─────────────────────────────────────────────────────────────────────────────

def _resize_for_efficientnet(spec: tf.Tensor, label: tf.Tensor):
    """
    Resize (N_MELS, T_FRAMES, 1) → (224, 224, 3) for EfficientNetB0.
    Grayscale → 3-channel by repeating the channel dim.
    """
    h, w, _ = EFFICIENTNET_INPUT_SIZE
    spec_3ch = tf.image.grayscale_to_rgb(spec)        # (N_MELS, T_FRAMES, 3)
    spec_resized = tf.image.resize(spec_3ch, [h, w])  # (224, 224, 3)
    # EfficientNet preprocessing (scale to [0, 255] range for preprocess_input)
    spec_scaled = (spec_resized + 1.0) * 127.5        # normalise from [-1,1] → [0,255]
    spec_scaled = tf.clip_by_value(spec_scaled, 0.0, 255.0)
    return spec_scaled, label


# ─────────────────────────────────────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    file_label_pairs: List[Tuple[str, int]],
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    cache: bool = False,
    repeat: bool = False,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    prefetch: int = tf.data.AUTOTUNE,
    model_type: str = "cnn",          # 'cnn' | 'cnn_lstm' | 'efficientnet'
    drop_remainder: bool = False,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset from (file_path, label) pairs.

    Args:
        file_label_pairs: List of (audio_file_path, label_int) tuples.
        batch_size:        Batch size.
        shuffle:           Shuffle the dataset (train only).
        cache:             Cache after first epoch (needs RAM or cache file).
        repeat:            Repeat indefinitely (for training).
        num_parallel_calls: Parallelism for map().
        prefetch:          How many batches to prefetch.
        model_type:        Adjusts output shape for 'efficientnet'.
        drop_remainder:    Drop the last incomplete batch.

    Returns:
        tf.data.Dataset yielding (spectrogram_batch, label_batch).
    """
    if not file_label_pairs:
        raise ValueError("file_label_pairs is empty — no data to build a dataset from.")

    paths  = [fp for fp, _ in file_label_pairs]
    labels = [lbl for _, lbl in file_label_pairs]

    path_ds  = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(labels, dtype=tf.int32)
    )
    dataset  = tf.data.Dataset.zip((path_ds, label_ds))

    # Shuffle before preprocessing so files are read in random order
    if shuffle:
        buffer_size = min(len(file_label_pairs), 10_000)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=SEED, reshuffle_each_iteration=True)

    # Preprocess
    dataset = dataset.map(_tf_preprocess, num_parallel_calls=num_parallel_calls)

    # EfficientNet post-processing (resize + rescale)
    if model_type == "efficientnet":
        dataset = dataset.map(_resize_for_efficientnet, num_parallel_calls=num_parallel_calls)

    if cache:
        dataset = dataset.cache()

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(prefetch)

    logger.info(
        "Built dataset: %d samples | batch=%d | shuffle=%s | model=%s",
        len(file_label_pairs), batch_size, shuffle, model_type,
    )
    return dataset


def build_train_dev_datasets(
    train_pairs: List[Tuple[str, int]],
    dev_pairs:   List[Tuple[str, int]],
    batch_size:  int = BATCH_SIZE,
    model_type:  str = "cnn",
    cache_train: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Convenience function: builds train and dev datasets with correct flags.

    Returns:
        (train_dataset, dev_dataset)
    """
    train_ds = build_dataset(
        train_pairs,
        batch_size=batch_size,
        shuffle=True,
        cache=cache_train,
        repeat=False,
        model_type=model_type,
    )
    dev_ds = build_dataset(
        dev_pairs,
        batch_size=batch_size,
        shuffle=False,
        cache=False,
        repeat=False,
        model_type=model_type,
    )
    return train_ds, dev_ds


def get_class_weights(labels: List[int]) -> dict:
    """
    Compute class weights for imbalanced datasets.
    Returns dict: {0: weight_for_spoof, 1: weight_for_bonafide}
    """
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=np.array(labels))
    return {int(c): float(w) for c, w in zip(classes, weights)}
