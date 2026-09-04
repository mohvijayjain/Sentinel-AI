import pandas as pd
from sqlalchemy import text

from src.database.postgres import engine


DATA_PATH = "data/reference/reference_data.parquet"
TABLE_NAME = "taxi_trips"


def load_taxi_data():
    print("Loading parquet file...")

    df = pd.read_parquet(DATA_PATH)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    print("Creating/loading PostgreSQL table...")

    # Replace existing table on first load
    df.to_sql(
        TABLE_NAME,
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=10_000,
        method="multi"
    )

    print("Taxi data loaded successfully.")


if __name__ == "__main__":
    load_taxi_data()