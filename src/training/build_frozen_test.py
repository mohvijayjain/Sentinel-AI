"""
Build data/frozen_test.parquet — ONCE.

    python -m src.training.build_frozen_test

This is the only place the evaluation set is defined. retrain.py reads the
file and never writes it. Re-running this script without --force refuses to
overwrite an existing frozen test set, because replacing it invalidates every
Champion/Challenger comparison already made against it.

Selection is a fixed-seed hash of the stable row_key. It does not look at
trip_duration, and it does not depend on row order, so re-running the
selection reproduces the identical row set.
"""

from __future__ import annotations

import argparse
import logging

from src.training.dataset import (
    FROZEN_TEST_PATH,
    FROZEN_TEST_SIZE,
    ROW_KEY,
    SEED,
    load_all_row_keys,
    load_rows_by_key,
    select_frozen_test_keys,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def build(force: bool = False) -> None:

    if FROZEN_TEST_PATH.exists() and not force:
        raise SystemExit(
            f"{FROZEN_TEST_PATH} already exists.\n"
            "Refusing to overwrite the frozen test set. Every metric ever "
            "compared against it assumes it never changes.\n"
            "Pass --force only if you deliberately intend to invalidate all "
            "previous Champion/Challenger comparisons."
        )

    logger.info("Reading row identity from processed Parquet...")

    all_keys = load_all_row_keys()

    logger.info("Corpus rows: %s", f"{len(all_keys):,}")

    if len(all_keys) != len(set(all_keys.tolist())):
        raise ValueError(
            "row_key is not unique across the processed files — "
            "the frozen test set cannot be defined safely."
        )

    test_keys = select_frozen_test_keys(all_keys, FROZEN_TEST_SIZE)

    logger.info(
        "Selected %s test rows (seed=%s)",
        f"{len(test_keys):,}",
        SEED,
    )

    frame = load_rows_by_key(test_keys)

    FROZEN_TEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame.to_parquet(FROZEN_TEST_PATH, index=False)

    logger.info("Frozen test set written -> %s", FROZEN_TEST_PATH)
    logger.info("Rows:     %s", f"{len(frame):,}")
    logger.info("Columns:  %s", list(frame.columns))
    logger.info("%s range: %s .. %s", ROW_KEY, frame[ROW_KEY].min(), frame[ROW_KEY].max())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing frozen test set (invalidates comparisons).",
    )

    build(force=parser.parse_args().force)
