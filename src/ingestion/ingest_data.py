import sys
from pathlib import Path

import pandas as pd

from src.ingestion.preprocess import clean_and_engineer, FEATURES
from src.database.postgres import engine


TABLE_NAME = "taxi_trips"


def ingest_file(file_path: str, month_name: str = ""):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"Loading: {path}")

    # Read raw dataset
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        raise ValueError("Only CSV and Parquet files are supported.")

    print(f"Raw rows: {len(df):,}")

    # Preprocess
    df = clean_and_engineer(
        df,
        month_name=month_name or path.stem
    )

    # Validate final schema
    missing = [col for col in FEATURES if col not in df.columns]

    if missing:
        raise ValueError(
            f"Processed data is missing columns: {missing}"
        )

    # Append processed data to PostgreSQL
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