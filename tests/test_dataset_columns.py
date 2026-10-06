"""
src/training/dataset.py column-case alignment.

preprocess.FEATURES switched to CamelCase (PULocationID, ...) while
frozen_test.parquet is stored lower-case and the loader still assumed a
lower-case canonical form, so every real retrain failed with
"frozen_test.parquet is missing required columns". Columns are now
matched case-insensitively and renamed to the FEATURES spelling: a pure
rename, data untouched.

Uses tiny temporary Parquet files; no real data needed.
"""

import numpy as np
import pandas as pd
import pytest

from src.training import dataset
from src.ingestion.preprocess import FEATURES, TARGET


REQUIRED = FEATURES + [TARGET]


def _frame(columns, rows=6, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.integers(1, 50, rows) for c in columns})


# ============================================================
# _canonical_names
# ============================================================

def test_lowercase_names_map_to_features_spelling():

    mapping = dataset._canonical_names([c.lower() for c in REQUIRED] + ["row_key"])

    assert set(mapping) == set(REQUIRED + ["row_key"])
    for canonical, ondisk in mapping.items():
        assert ondisk == canonical.lower()


def test_features_spelling_maps_to_itself():

    mapping = dataset._canonical_names(REQUIRED)

    assert mapping == {c: c for c in REQUIRED}


def test_absent_columns_are_left_out():

    mapping = dataset._canonical_names(["trip_distance", "__index_level_0__"])

    assert mapping == {"trip_distance": "trip_distance"}


def test_case_only_duplicates_are_rejected():

    with pytest.raises(ValueError, match="Ambiguous"):
        dataset._canonical_names(["PULocationID", "pulocationid"])


def test_features_contain_mixed_case_columns():
    """The situation the fix exists for: FEATURES is not all lower-case."""

    assert any(c != c.lower() for c in FEATURES)


# ============================================================
# load_frozen_test: lower-case file, FEATURES-spelled frame
# ============================================================

def test_frozen_test_lowercase_file_loads(tmp_path, monkeypatch):

    stored = _frame([c.lower() for c in REQUIRED] + ["row_key"])
    path = tmp_path / "frozen_test.parquet"
    stored.to_parquet(path)
    monkeypatch.setattr(dataset, "FROZEN_TEST_PATH", path)

    loaded = dataset.load_frozen_test()

    for column in REQUIRED + ["row_key"]:
        assert column in loaded.columns

    # Pure rename: identical values, same order
    assert np.array_equal(loaded.to_numpy(), stored.to_numpy())


def test_frozen_test_still_reports_truly_missing_columns(tmp_path, monkeypatch):

    stored = _frame([c.lower() for c in REQUIRED if c != TARGET] + ["row_key"])
    path = tmp_path / "frozen_test.parquet"
    stored.to_parquet(path)
    monkeypatch.setattr(dataset, "FROZEN_TEST_PATH", path)

    with pytest.raises(ValueError, match="missing required columns"):
        dataset.load_frozen_test()


# ============================================================
# load_rows_by_key: CamelCase processed files
# ============================================================

def test_rows_by_key_from_camelcase_processed_files(tmp_path, monkeypatch):

    files = {}
    for i, month in enumerate((202601, 202602)):
        path = tmp_path / f"m{month}.parquet"
        _frame(REQUIRED, rows=5, seed=i).to_parquet(path)
        files[month] = path

    monkeypatch.setattr(dataset, "SOURCE_FILES", files)

    keys = dataset.load_all_row_keys()
    rows = dataset.load_rows_by_key(keys[:4])

    assert list(rows.columns) == REQUIRED + ["row_key"]
    assert len(rows) == 4


def test_rows_by_key_from_lowercase_processed_files(tmp_path, monkeypatch):
    """Either on-disk spelling works; output always uses FEATURES."""

    path = tmp_path / "m202601.parquet"
    _frame([c.lower() for c in REQUIRED], rows=5).to_parquet(path)
    monkeypatch.setattr(dataset, "SOURCE_FILES", {202601: path})

    rows = dataset.load_rows_by_key(dataset.load_all_row_keys()[:3])

    assert list(rows.columns) == REQUIRED + ["row_key"]
