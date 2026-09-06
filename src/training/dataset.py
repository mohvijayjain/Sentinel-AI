"""
Single source of truth for training / evaluation data.

Training and evaluation data live in Parquet:

    data/raw/*.parquet
        -> data/processed/*_clean.parquet   (via src.ingestion.preprocess)
        -> training sample + data/frozen_test.parquet

PostgreSQL is no longer queried for the large training sample. It remains
responsible for monitoring runs, drift scores and model decision metadata.

Row identity
------------
The processed Parquet files preserve the original pandas index (the raw row
position that survived cleaning), which is unique within each monthly file.
``row_key = month * 2**32 + original_index`` therefore forms a stable,
deterministic primary key across the whole training corpus, without needing
to hash feature values (which would collide on genuinely duplicate trips).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.preprocess import FEATURES, TARGET


# ── Paths ──────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")

FROZEN_TEST_PATH = Path("data/frozen_test.parquet")

# Months that make up the training corpus. April is reserved as the drift
# / "current" dataset and is deliberately excluded.
SOURCE_FILES = {
    202601: PROCESSED_DIR / "yellow_tripdata_2026-01_clean.parquet",
    202602: PROCESSED_DIR / "yellow_tripdata_2026-02_clean.parquet",
    202603: PROCESSED_DIR / "yellow_tripdata_2026-03_clean.parquet",
}

ROW_KEY = "row_key"

FROZEN_TEST_SIZE = 200_000

# Fixed seed. Changing this changes the frozen test set and invalidates every
# comparison ever made against it.
SEED = 42

# The processed Parquet files on disk were written before preprocess.py
# started lower-casing column names, so they still carry the original CamelCase
# for these four. This is exactly the aliasing the previous SQL loader did
# (`"PULocationID" AS pulocationid`, ...) — a pure rename, no feature logic.
COLUMN_ALIASES = {
    "PULocationID": "pulocationid",
    "DOLocationID": "dolocationid",
    "VendorID": "vendorid",
    "RatecodeID": "ratecodeid",
}

REQUIRED_COLUMNS = FEATURES + [TARGET]


# ── Deterministic hashing ──────────────────────────────
def _splitmix64(values: np.ndarray) -> np.ndarray:
    """
    Vectorised splitmix64.

    Used to assign every row a stable pseudo-random ordinal that depends only
    on its row_key — never on row order, file order, or trip_duration. Pure
    integer arithmetic, so it is reproducible across platforms, Python
    versions and NumPy versions (unlike ``hash()`` or ``DataFrame.sample``
    over a re-ordered frame).
    """

    x = values.astype(np.uint64, copy=True) ^ np.uint64(SEED)

    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)

    return x ^ (x >> np.uint64(31))


def _row_keys(month: int, index: pd.Index) -> np.ndarray:
    return (np.int64(month) << np.int64(32)) + index.to_numpy(dtype=np.int64)


def _ondisk_names(path: Path) -> dict[str, str]:
    """Map canonical (lowercase) column name -> the name used on disk."""

    import pyarrow.parquet as pq

    names = pq.ParquetFile(path).schema_arrow.names

    return {
        COLUMN_ALIASES.get(name, name.lower()): name
        for name in names
    }


def _check_source_files() -> None:
    missing = [str(p) for p in SOURCE_FILES.values() if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Processed training Parquet is missing: "
            f"{missing}. Generate it with "
            "`python -m src.ingestion.preprocess` before training."
        )


# ── Public API ─────────────────────────────────────────
def load_all_row_keys() -> np.ndarray:
    """
    Read only the row identity of the whole training corpus.

    Cheap: reads a single narrow column per file, so the full 7M-row corpus
    never has to be materialised just to decide which rows to use.
    """

    _check_source_files()

    keys = []

    for month, path in SOURCE_FILES.items():
        ondisk = _ondisk_names(path)
        frame = pd.read_parquet(path, columns=[ondisk["pickup_hour"]])
        keys.append(_row_keys(month, frame.index))

    return np.concatenate(keys)


def load_rows_by_key(wanted: np.ndarray) -> pd.DataFrame:
    """
    Load exactly the rows whose row_key appears in ``wanted``.

    Reads one month at a time and keeps only the selected slice, so peak
    memory stays bounded by a single monthly file rather than the whole
    corpus.

    Returns a frame with columns FEATURES + [TARGET] + [ROW_KEY], with the
    feature names in the exact lowercase form the model expects.
    """

    _check_source_files()

    wanted = np.unique(np.asarray(wanted, dtype=np.int64))

    chunks = []

    for month, path in SOURCE_FILES.items():
        ondisk = _ondisk_names(path)

        missing = [c for c in REQUIRED_COLUMNS if c not in ondisk]

        if missing:
            raise ValueError(
                f"{path} is missing required columns: {missing}"
            )

        index_frame = pd.read_parquet(
            path,
            columns=[ondisk["pickup_hour"]],
        )

        keys = _row_keys(month, index_frame.index)
        del index_frame

        mask = np.isin(keys, wanted, assume_unique=True)

        if not mask.any():
            continue

        frame = pd.read_parquet(
            path,
            columns=[ondisk[c] for c in REQUIRED_COLUMNS],
        )

        frame = frame.rename(
            columns={ondisk[c]: c for c in REQUIRED_COLUMNS}
        )

        frame = frame.loc[mask].copy()
        frame[ROW_KEY] = keys[mask]

        chunks.append(frame.reset_index(drop=True))

    if not chunks:
        raise ValueError("No rows matched the requested row keys.")

    result = pd.concat(chunks, ignore_index=True)

    found = len(result)

    if found != len(wanted):
        raise ValueError(
            f"Requested {len(wanted):,} rows but found {found:,}. "
            "The processed Parquet files no longer contain the expected "
            "rows — data/frozen_test.parquet was built against different "
            "source data and is no longer valid."
        )

    return result[REQUIRED_COLUMNS + [ROW_KEY]]


def select_frozen_test_keys(
    all_keys: np.ndarray,
    size: int = FROZEN_TEST_SIZE,
) -> np.ndarray:
    """
    Choose the frozen test rows.

    Selection depends only on row_key through a fixed-seed hash — it never
    looks at trip_duration, and it is independent of row order, so re-running
    it yields a byte-identical set.
    """

    all_keys = np.asarray(all_keys, dtype=np.int64)

    ordinals = _splitmix64(all_keys)

    # Sort by hash, tie-break on the key itself so the result is total.
    order = np.lexsort((all_keys, ordinals))

    return np.sort(all_keys[order[:size]])


def load_frozen_test() -> pd.DataFrame:
    """
    Read THE evaluation dataset, shared by Champion and Challenger.

    Never creates or regenerates the file. If it is absent this raises, by
    design: silently rebuilding it would invalidate every metric previously
    compared against it.
    """

    if not FROZEN_TEST_PATH.exists():
        raise FileNotFoundError(
            f"Frozen test set not found at {FROZEN_TEST_PATH}.\n"
            "It is NOT regenerated automatically, because a regenerated test "
            "set silently invalidates every Champion/Challenger comparison "
            "made against the old one.\n"
            "Create it once with: python -m src.training.build_frozen_test"
        )

    frame = pd.read_parquet(FROZEN_TEST_PATH)

    missing = [
        column
        for column in REQUIRED_COLUMNS + [ROW_KEY]
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{FROZEN_TEST_PATH} is missing required columns: {missing}"
        )

    return frame
