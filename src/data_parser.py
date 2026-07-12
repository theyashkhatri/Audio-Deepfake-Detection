"""
DeepShield Audio — Data Parser
================================
Parses ASVspoof 2019 LA official CM protocol files to produce
clean (file_path, label) tuples for each split.

Protocol file format (space-separated):
  SPEAKER_ID  FILE_ID  -  SYSTEM_ID  LABEL
  e.g.: LA_0001  LA_T_1000137  -  -  bonafide
        LA_0001  LA_T_1000265  -  A17  spoof

No cross-split contamination: we only use files listed in the
official protocol for each split. Train ↔ Dev ↔ Eval are disjoint.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import pandas as pd

# Adjust import based on execution context
try:
    from src.config import (
        TRAIN_DIR, DEV_DIR, EVAL_DIR,
        TRAIN_PROTOCOL, DEV_PROTOCOL, EVAL_PROTOCOL,
        LABEL_MAP,
    )
except ImportError:
    from config import (
        TRAIN_DIR, DEV_DIR, EVAL_DIR,
        TRAIN_PROTOCOL, DEV_PROTOCOL, EVAL_PROTOCOL,
        LABEL_MAP,
    )

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_protocol_file(protocol_path: Path) -> pd.DataFrame:
    """
    Parse a single ASVspoof2019 protocol file into a DataFrame.

    Returns columns: speaker_id, file_id, system_id, label_str, label_int
    """
    if not protocol_path.exists():
        raise FileNotFoundError(
            f"Protocol file not found: {protocol_path}\n"
            "Please download the ASVspoof2019 LA dataset and place it under data/ASVspoof2019_LA/"
        )

    records = []
    with open(protocol_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                logger.warning("Skipping malformed line: %s", line)
                continue
            speaker_id = parts[0]
            file_id    = parts[1]
            # parts[2] is always '-' (unused)
            system_id  = parts[3]   # '-' for bonafide, 'A01'–'A19' for spoof
            label_str  = parts[4].lower()  # 'bonafide' or 'spoof'

            if label_str not in LABEL_MAP:
                logger.warning("Unknown label '%s' in line: %s", label_str, line)
                continue

            records.append({
                "speaker_id": speaker_id,
                "file_id":    file_id,
                "system_id":  system_id,
                "label_str":  label_str,
                "label":      LABEL_MAP[label_str],
            })

    df = pd.DataFrame(records)
    logger.info("Parsed %d records from %s", len(df), protocol_path.name)
    return df


def _attach_paths(df: pd.DataFrame, audio_dir: Path) -> pd.DataFrame:
    """
    Attach the absolute audio file path (.flac) to each row.
    Filters out rows where the file does not exist on disk.
    """
    df = df.copy()
    df["file_path"] = df["file_id"].apply(
        lambda fid: str(audio_dir / f"{fid}.flac")
    )

    exists_mask = df["file_path"].apply(lambda p: Path(p).exists())
    missing = (~exists_mask).sum()
    if missing > 0:
        logger.warning(
            "%d / %d files not found on disk in %s — skipping them.",
            missing, len(df), audio_dir
        )
    return df[exists_mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def load_split(
    split: str,
    protocol_path: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load a single split (train / dev / eval) into a DataFrame.

    Args:
        split: One of 'train', 'dev', 'eval'.
        protocol_path: Override the default protocol file path.
        audio_dir: Override the default audio directory.

    Returns:
        DataFrame with columns:
            speaker_id, file_id, system_id, label_str, label, file_path
    """
    split = split.lower()
    _defaults = {
        "train": (TRAIN_PROTOCOL, TRAIN_DIR),
        "dev":   (DEV_PROTOCOL,   DEV_DIR),
        "eval":  (EVAL_PROTOCOL,  EVAL_DIR),
    }
    if split not in _defaults:
        raise ValueError(f"split must be 'train', 'dev', or 'eval'. Got: {split}")

    proto, adir = _defaults[split]
    protocol_path = protocol_path or proto
    audio_dir     = audio_dir     or adir

    df = _parse_protocol_file(protocol_path)
    df = _attach_paths(df, audio_dir)
    df["split"] = split
    return df


def load_all_splits(
    train_protocol: Optional[Path] = None,
    dev_protocol: Optional[Path] = None,
    eval_protocol: Optional[Path] = None,
    train_dir: Optional[Path] = None,
    dev_dir: Optional[Path] = None,
    eval_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load all three official splits.

    Returns:
        Dict with keys 'train', 'dev', 'eval', each containing a DataFrame.
    """
    return {
        "train": load_split("train", train_protocol, train_dir),
        "dev":   load_split("dev",   dev_protocol,   dev_dir),
        "eval":  load_split("eval",  eval_protocol,  eval_dir),
    }


def get_file_label_pairs(df: pd.DataFrame) -> List[Tuple[str, int]]:
    """
    Extract (file_path, label) tuples from a split DataFrame.

    Returns:
        List of (absolute_file_path_str, label_int) tuples.
        label_int: 1 = bonafide (real), 0 = spoof (fake)
    """
    return list(zip(df["file_path"], df["label"]))


def get_class_distribution(df: pd.DataFrame) -> pd.Series:
    """Return label_str value counts for EDA."""
    return df["label_str"].value_counts()


def get_system_distribution(df: pd.DataFrame) -> pd.Series:
    """Return spoof system_id breakdown for EDA."""
    spoof_df = df[df["label_str"] == "spoof"]
    return spoof_df["system_id"].value_counts()


def verify_no_leakage(splits: Dict[str, pd.DataFrame]) -> bool:
    """
    Verify that train / dev / eval file_ids are strictly disjoint.

    Raises AssertionError if any overlap is detected.
    Returns True if clean.
    """
    train_ids = set(splits["train"]["file_id"])
    dev_ids   = set(splits["dev"]["file_id"])
    eval_ids  = set(splits["eval"]["file_id"])

    td_overlap = train_ids & dev_ids
    te_overlap = train_ids & eval_ids
    de_overlap = dev_ids   & eval_ids

    assert len(td_overlap) == 0, f"Train/Dev overlap: {len(td_overlap)} files!"
    assert len(te_overlap) == 0, f"Train/Eval overlap: {len(te_overlap)} files!"
    assert len(de_overlap) == 0, f"Dev/Eval overlap: {len(de_overlap)} files!"

    logger.info("✅ No data leakage detected. All splits are disjoint.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# DEMO / SYNTHETIC DATA UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_dataframe(
    n_bonafide: int = 50,
    n_spoof: int = 100,
    audio_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Create a synthetic DataFrame for pipeline testing without the real dataset.
    Generates dummy paths (files won't exist on disk).

    Args:
        n_bonafide: Number of bonafide samples.
        n_spoof: Number of spoof samples.
        audio_dir: Base directory (informational only).

    Returns:
        DataFrame mimicking the real protocol output.
    """
    import numpy as np
    rng = np.random.default_rng(42)
    audio_dir = audio_dir or Path("/tmp/fake_audio")

    records = []
    systems = ["A01", "A02", "A03", "A04", "A05", "A06", "A07"]

    for i in range(n_bonafide):
        fid = f"LA_SYNTH_REAL_{i:06d}"
        records.append({
            "speaker_id": f"LA_{rng.integers(1, 20):04d}",
            "file_id":    fid,
            "system_id":  "-",
            "label_str":  "bonafide",
            "label":      1,
            "file_path":  str(audio_dir / f"{fid}.flac"),
            "split":      "synthetic",
        })

    for i in range(n_spoof):
        fid = f"LA_SYNTH_FAKE_{i:06d}"
        records.append({
            "speaker_id": f"LA_{rng.integers(1, 20):04d}",
            "file_id":    fid,
            "system_id":  systems[i % len(systems)],
            "label_str":  "spoof",
            "label":      0,
            "file_path":  str(audio_dir / f"{fid}.flac"),
            "split":      "synthetic",
        })

    return pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
