# src/ingestion/preprocess.py

import pandas as pd
import numpy as np
import os
import gc

# ── Constants ──────────────────────────────────────────
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
REFERENCE_DIR = "data/reference"

TARGET = "trip_duration"


FEATURES = [
    "trip_distance",
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "is_weekend",
    "is_rush_hour",
    "pulocationid",
    "dolocationid",
    "payment_type",
    "vendorid",
    "ratecodeid",
]

DROP_COLS = [
    "passenger_count",
    "store_and_fwd_flag",
    "congestion_surcharge",
    "Airport_fee",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "tip_amount",
    "tolls_amount",
    "total_amount",
    "improvement_surcharge",
    "mta_tax",
    "extra",
    "fare_amount",          # post-trip leakage
]

# ── Step 1: Convert Datetime ───────────────────────────
def convert_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df["tpep_pickup_datetime"]  = pd.to_datetime(
        df["tpep_pickup_datetime"],  errors="coerce"
    )
    df["tpep_dropoff_datetime"] = pd.to_datetime(
        df["tpep_dropoff_datetime"], errors="coerce"
    )
    # Drop rows where datetime conversion failed
    df = df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime"])
    return df

# ── Step 2: Create Target Variable ────────────────────
def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df["trip_duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds()
    return df

# ── Step 3: Feature Engineering ───────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["pickup_hour"]        = df["tpep_pickup_datetime"].dt.hour
    df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["pickup_month"]       = df["tpep_pickup_datetime"].dt.month
    df["is_weekend"]         = df["pickup_day_of_week"].isin([5, 6]).astype(int)
    df["is_rush_hour"]       = df["pickup_hour"].isin(
                                [7, 8, 9, 17, 18, 19]
                            ).astype(int)
    return df

# ── Step 4: Remove Outliers ────────────────────────────
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # ── trip_distance ──────────────────────────────────
    print(f"\n  Before distance filter:    {len(df):,}")
    df = df[df["trip_distance"] > 0.5]
    df = df[df["trip_distance"] <= 30]
    print(f"  After distance filter:     {len(df):,}")

    # ── trip_duration ──────────────────────────────────
    print(f"\n  Before duration filter:    {len(df):,}")
    df = df[df["trip_duration"] >= 60]
    df = df[df["trip_duration"] <= 7200]
    print(f"  After duration filter:     {len(df):,}")

    # ── RatecodeID ─────────────────────────────────────


    # ── PULocationID ───────────────────────────────────
    if "PULocationID" in df.columns:
        print(f"\n  Before PULocationID filter:{len(df):,}")
        df = df[df["PULocationID"].between(1, 263)]
        print(f"  After PULocationID filter: {len(df):,}")

    # ── DOLocationID ───────────────────────────────────
    if "DOLocationID" in df.columns:
        print(f"\n  Before DOLocationID filter:{len(df):,}")
        df = df[df["DOLocationID"].between(1, 263)]
        print(f"  After DOLocationID filter: {len(df):,}")

    # ── payment_type ───────────────────────────────────


    # ── Summary ────────────────────────────────────────
    after       = len(df)
    removed     = before - after
    removed_pct = removed / before * 100
    print(f"\n  {'='*35}")
    print(f"  Total removed: {removed:,} ({removed_pct:.1f}%)")
    print(f"  Rows kept:     {after:,} ({100-removed_pct:.1f}%)")
    print(f"  {'='*35}")

    return df

# ── Step 5: Drop Unnecessary Columns ──────────────────
def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=DROP_COLS, errors="ignore")
    return df

# ── Step 6: Select Final Features ─────────────────────
def select_features(df: pd.DataFrame) -> pd.DataFrame:
    required = FEATURES + [TARGET]
    existing = [f for f in required if f in df.columns]
    df = df[existing].dropna()
    return df

# ── Step 7: Optimize RAM ───────────────────────────────
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes("int64").columns:
        df[col] = df[col].astype("int32")
    return df

# ── Master Clean Function ──────────────────────────────
def clean_and_engineer(df: pd.DataFrame, month_name: str = "") -> pd.DataFrame:

    print(f"\n{'='*45}")
    print(f"  Processing {month_name}")
    print(f"{'='*45}")
    print(f"  Raw rows:         {len(df):,}")

    df = convert_datetime(df)
    print(f"  After datetime:   {len(df):,}")

    df = create_target(df)
    df = engineer_features(df)
    df = remove_outliers(df)
    df = drop_columns(df)
    df = select_features(df)
    df = optimize_dtypes(df)

    print(f"  Final rows:       {len(df):,}")
    print(f"  RAM usage:        {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
    print(f"  Avg duration:     {df['trip_duration'].mean()/60:.1f} minutes")
    print(f"  Features:         {list(df.columns)}")

    df.columns = df.columns.str.lower()

    return df


# ── Main Pipeline ──────────────────────────────────────
if __name__ == "__main__":

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REFERENCE_DIR, exist_ok=True)

    # ── Process Jan, Feb, Mar (Training Data) ─────────
    TRAIN_MONTHS = {1: "January", 2: "February", 3: "March"}

    for month, name in TRAIN_MONTHS.items():
        raw_path = f"{RAW_DIR}/yellow_tripdata_2026-{month:02d}.parquet"
        save_path = f"{PROCESSED_DIR}/yellow_tripdata_2026-{month:02d}_clean.parquet"
        df = pd.read_parquet(raw_path)
        df = clean_and_engineer(df, month_name=name)
        df.to_parquet(save_path)
        print(f"  ✅ Saved → {save_path}")

        del df
        gc.collect()

    # ── Combine into Reference Dataset ────────────────
    print(f"\n{'='*45}")
    print("  Building Reference Dataset")
    print(f"{'='*45}")

    dfs = []
    for month, name in TRAIN_MONTHS.items():
        path = f"{PROCESSED_DIR}/yellow_tripdata_2026-{month:02d}_clean.parquet"
        df   = pd.read_parquet(path)
        dfs.append(df)
        print(f"  {name}: {len(df):,} rows")

    reference_df = pd.concat(dfs, ignore_index=True)
    reference_df.to_parquet(f"{REFERENCE_DIR}/reference_data.parquet")

    print(f"\n  ✅ Reference data saved!")
    print(f"  Total rows:       {len(reference_df):,}")
    print(f"  Total RAM:        {reference_df.memory_usage(deep=True).sum()/1e6:.1f} MB")
    print(f"  Avg duration:     {reference_df['trip_duration'].mean()/60:.1f} minutes")
    print(f"  Min duration:     {reference_df['trip_duration'].min()/60:.1f} minutes")
    print(f"  Max duration:     {reference_df['trip_duration'].max()/60:.1f} minutes")
    print(f"  Features:         {[c for c in reference_df.columns if c != 'trip_duration']}")
    print(f"  Target:           trip_duration (seconds)")

    del dfs, reference_df
    gc.collect()

    # ── Process April (Drift Detection Data) ──────────
    print(f"\n{'='*45}")
    print("  Processing April (Drift Data)")
    print(f"{'='*45}")

    apr = pd.read_parquet(f"{RAW_DIR}/taxi_2024_04.parquet")
    apr = clean_and_engineer(apr, month_name="April")
    apr.to_parquet(f"{PROCESSED_DIR}/april_drift_data.parquet")
    print(f"  ✅ April saved → {PROCESSED_DIR}/april_drift_data.parquet")

    del apr
    gc.collect()

    # ── Final Summary ──────────────────────────────────
    print(f"\n{'='*45}")
    print("  ALL PREPROCESSING COMPLETE!")
    print(f"{'='*45}")
