"""
Tests for src/data_parser.py

Uses temporary protocol files and synthetic audio to test parsing
without requiring the full ASVspoof2019 dataset.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_parser import (
    _parse_protocol_file,
    _attach_paths,
    get_file_label_pairs,
    get_class_distribution,
    get_system_distribution,
    verify_no_leakage,
    make_synthetic_dataframe,
)
from src.config import LABEL_MAP


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

PROTOCOL_CONTENT = """\
LA_0001 LA_T_0000001 - - bonafide
LA_0001 LA_T_0000002 - A01 spoof
LA_0002 LA_T_0000003 - A02 spoof
LA_0002 LA_T_0000004 - - bonafide
LA_0003 LA_T_0000005 - A07 spoof
"""


@pytest.fixture
def protocol_file(tmp_path):
    """Write a minimal mock protocol file."""
    p = tmp_path / "mock_protocol.txt"
    p.write_text(PROTOCOL_CONTENT)
    return p


@pytest.fixture
def audio_dir_with_files(tmp_path):
    """Create dummy .flac files matching the protocol."""
    import soundfile as sf
    import numpy as np

    audio_dir = tmp_path / "flac"
    audio_dir.mkdir()
    file_ids = ["LA_T_0000001", "LA_T_0000002", "LA_T_0000003",
                "LA_T_0000004", "LA_T_0000005"]
    for fid in file_ids:
        wav = (np.sin(np.linspace(0, np.pi, 1600)) * 0.3).astype(np.float32)
        sf.write(str(audio_dir / f"{fid}.flac"), wav, 16000, subtype="PCM_16",
                 format="FLAC")
    return audio_dir


# ─────────────────────────────────────────────────────────────────────────────
# _parse_protocol_file()
# ─────────────────────────────────────────────────────────────────────────────

class TestParseProtocolFile:
    def test_returns_dataframe(self, protocol_file):
        df = _parse_protocol_file(protocol_file)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self, protocol_file):
        df = _parse_protocol_file(protocol_file)
        assert len(df) == 5

    def test_expected_columns(self, protocol_file):
        df = _parse_protocol_file(protocol_file)
        for col in ["speaker_id", "file_id", "system_id", "label_str", "label"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_label_encoding(self, protocol_file):
        df = _parse_protocol_file(protocol_file)
        bonafide = df[df["label_str"] == "bonafide"]["label"].unique()
        spoof    = df[df["label_str"] == "spoof"]["label"].unique()
        assert list(bonafide) == [LABEL_MAP["bonafide"]]
        assert list(spoof)    == [LABEL_MAP["spoof"]]

    def test_system_id_bonafide(self, protocol_file):
        df = _parse_protocol_file(protocol_file)
        bonafide_rows = df[df["label_str"] == "bonafide"]
        assert all(bonafide_rows["system_id"] == "-")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _parse_protocol_file(tmp_path / "does_not_exist.txt")

    def test_malformed_line_skipped(self, tmp_path):
        p = tmp_path / "bad_protocol.txt"
        p.write_text("LA_0001 LA_T_0001\nLA_0002 LA_T_0002 - - bonafide\n")
        df = _parse_protocol_file(p)
        assert len(df) == 1   # Only the valid line is kept


# ─────────────────────────────────────────────────────────────────────────────
# _attach_paths()
# ─────────────────────────────────────────────────────────────────────────────

class TestAttachPaths:
    def test_filters_missing_files(self, protocol_file, tmp_path):
        df = _parse_protocol_file(protocol_file)
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        result = _attach_paths(df, empty_dir)
        assert len(result) == 0, "All files missing — DataFrame should be empty"

    def test_found_files_included(self, protocol_file, audio_dir_with_files):
        df     = _parse_protocol_file(protocol_file)
        result = _attach_paths(df, audio_dir_with_files)
        assert len(result) == 5
        assert "file_path" in result.columns

    def test_paths_are_absolute_strings(self, protocol_file, audio_dir_with_files):
        df     = _parse_protocol_file(protocol_file)
        result = _attach_paths(df, audio_dir_with_files)
        for p in result["file_path"]:
            assert isinstance(p, str)
            assert Path(p).is_absolute()


# ─────────────────────────────────────────────────────────────────────────────
# get_file_label_pairs()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetFileLabelPairs:
    def test_returns_list_of_tuples(self):
        df = make_synthetic_dataframe(n_bonafide=10, n_spoof=20)
        pairs = get_file_label_pairs(df)
        assert isinstance(pairs, list)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)

    def test_correct_count(self):
        df    = make_synthetic_dataframe(n_bonafide=5, n_spoof=10)
        pairs = get_file_label_pairs(df)
        assert len(pairs) == 15

    def test_labels_are_binary(self):
        df    = make_synthetic_dataframe()
        pairs = get_file_label_pairs(df)
        labels = [lbl for _, lbl in pairs]
        assert set(labels) <= {0, 1}


# ─────────────────────────────────────────────────────────────────────────────
# get_class_distribution()
# ─────────────────────────────────────────────────────────────────────────────

class TestClassDistribution:
    def test_correct_counts(self):
        df   = make_synthetic_dataframe(n_bonafide=30, n_spoof=70)
        dist = get_class_distribution(df)
        assert dist["bonafide"] == 30
        assert dist["spoof"]    == 70

    def test_returns_series(self):
        df   = make_synthetic_dataframe()
        dist = get_class_distribution(df)
        assert isinstance(dist, pd.Series)


# ─────────────────────────────────────────────────────────────────────────────
# verify_no_leakage()
# ─────────────────────────────────────────────────────────────────────────────

class TestVerifyNoLeakage:
    def test_clean_splits_pass(self):
        """Disjoint file IDs → no leakage."""
        import numpy as np

        def _make_split(offset, n=50):
            n_bon = n // 5
            n_spf = n - n_bon
            return pd.DataFrame({
                "file_id":   [f"LA_T_{offset + i:07d}" for i in range(n)],
                "label_str": ["bonafide"] * n_bon + ["spoof"] * n_spf,
                "label":     [1] * n_bon + [0] * n_spf,
            })

        splits = {
            "train": _make_split(0,   n=100),
            "dev":   _make_split(100, n=50),
            "eval":  _make_split(150, n=50),
        }
        assert verify_no_leakage(splits) is True

    def test_overlapping_splits_raise(self):
        """Overlapping file IDs → AssertionError."""
        shared = pd.DataFrame({"file_id": ["LA_T_0000001", "LA_T_0000002"],
                               "label_str": ["bonafide", "spoof"],
                               "label": [1, 0]})
        splits = {"train": shared, "dev": shared, "eval": shared}
        with pytest.raises(AssertionError):
            verify_no_leakage(splits)


# ─────────────────────────────────────────────────────────────────────────────
# make_synthetic_dataframe()
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeSyntheticDataframe:
    def test_correct_total_count(self):
        df = make_synthetic_dataframe(n_bonafide=20, n_spoof=40)
        assert len(df) == 60

    def test_expected_columns(self):
        df = make_synthetic_dataframe()
        for col in ["speaker_id", "file_id", "system_id", "label_str", "label", "file_path"]:
            assert col in df.columns

    def test_label_values(self):
        df = make_synthetic_dataframe(n_bonafide=10, n_spoof=10)
        assert set(df["label"].unique()) == {0, 1}

    def test_unique_file_ids(self):
        df = make_synthetic_dataframe()
        assert df["file_id"].nunique() == len(df)
