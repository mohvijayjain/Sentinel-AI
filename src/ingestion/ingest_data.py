import sys
from pathlib import Path

import pandas as pd

from src.ingestion.preprocess import clean_and_engineer, FEATURES


TABLE_NAME = "taxi_trips"

SUPPORTED_SUFFIXES = (".csv", ".parquet", ".pq")

# Raw NYC TLC columns clean_and_engineer() needs to produce every model
# FEATURE (and the trip_duration target), with TLC's exact names/case.
# Everything else in a raw file is dropped by preprocessing anyway.
RAW_REQUIRED_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "VendorID",
    "RatecodeID",
]


def read_raw_file(path, columns=None) -> pd.DataFrame:
    """Read a raw CSV / Parquet file (optionally only `columns`)."""

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, usecols=columns)

    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path, columns=columns)

    raise ValueError("Only CSV and Parquet files are supported.")


def preprocess_raw(df: pd.DataFrame, month_name: str = "") -> pd.DataFrame:
    """Shared preprocessing, then the processed-schema check."""

    df = clean_and_engineer(df, month_name=month_name)

    missing = [col for col in FEATURES if col not in df.columns]

    if missing:
        raise ValueError(
            f"Processed data is missing columns: {missing}"
        )

    return df


def ingest_file(file_path: str, month_name: str = ""):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"Loading: {path}")

    # Read raw dataset
    df = read_raw_file(path)

    print(f"Raw rows: {len(df):,}")

    # Preprocess + validate final schema
    df = preprocess_raw(df, month_name=month_name or path.stem)

    # Append processed data to PostgreSQL
    from src.database.postgres import engine

    print(f"Appending {len(df):,} rows to PostgreSQL...")

    df.to_sql(
        TABLE_NAME,
        engine,
        if_exists="append",
        index=False,
        chunksize=10_000,
        method="multi",
    )

    print("✅ Data successfully appended to PostgreSQL.")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.ingestion.ingest_data "
            "<file_path> [month_name]"
        )
        sys.exit(1)

    file_path = sys.argv[1]
    month_name = sys.argv[2] if len(sys.argv) > 2 else ""

    ingest_file(file_path, month_name)
