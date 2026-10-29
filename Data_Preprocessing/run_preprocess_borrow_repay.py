"""
Preprocessing script for the borrow_repay dataset (FinSurvival).

Raw data preparation (prerequisite):
  Run create_borrow_repay_csv.R from the root of the FinSurvival repository
  (https://github.com/Large-Transaction-Models/DMLR_DeFi_Survival_Benchmark),
  then place the output at:
    Data_Preprocessing/data/Finsurvival/borrow_repay/borrow_repay_first_obs.csv

The CSV already has num_/fac_ prefixes on all feature columns plus time/event.
See configs/borrow_repay.yaml for the full reference.

Usage: python run_preprocess_borrow_repay.py <method_name>
"""
import pandas as pd
import yaml
import os
import sys
from src.utils_preprocessing import standard_pipeline

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_NAME = "borrow_repay"
CONFIG_PATH  = os.path.join(SCRIPT_DIR, "configs", f"{DATASET_NAME}.yaml")
RAW_PATH = os.path.join(
    SCRIPT_DIR, "data", "Finsurvival", "borrow_repay",
    "borrow_repay_first_obs.csv",
)
OUT_DIR  = os.path.join(SCRIPT_DIR, "data", "processed", "borrow_repay")
OUT_PATH = os.path.join(OUT_DIR, "borrow_repay_processed.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_preprocess_borrow_repay.py <method_name>")
        print("  method_name: name of method config in ../Model_Training/method_configs/")
        sys.exit(1)

    method_name = sys.argv[1]

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    TIME_COL  = config.get("TIME_COL", "time")
    EVENT_COL = config.get("EVENT_COL", "event")

    method_config_path = os.path.join(
        SCRIPT_DIR, "..", "Model_Training", "method_configs", f"{method_name}.yaml"
    )
    if not os.path.exists(method_config_path):
        print(f"Error: Method config '{method_config_path}' not found")
        sys.exit(1)

    with open(method_config_path) as f:
        method_config = yaml.safe_load(f)

    scaling        = method_config.get("scaling", True)
    drop_first     = method_config.get("drop_first", True)
    dropNZV        = method_config.get("dropNZV", True)
    drop_high_corr = method_config.get("drop_high_corr", True)
    min_freq       = method_config.get("min_freq", 0.0025)
    corr_threshold = method_config.get("corr_threshold", 0.98)

    print("-" * 50)
    print(f"\tDataset:{DATASET_NAME}")
    print(f"\tMethod:{method_name}")
    print(f"\tTime col:{TIME_COL}")
    print(f"\tEvent col:{EVENT_COL}")
    print(f"\tScaling:{scaling}")
    print(f"\tDrop first OHE cat:{drop_first}")
    print(f"\tDrop NZV:{dropNZV} (min_freq={min_freq})")
    print(f"\tDrop high corr:{drop_high_corr} (threshold={corr_threshold})")
    print("-" * 50)

    if not os.path.exists(RAW_PATH):
        print(f"Error: Raw data not found at {RAW_PATH}")
        sys.exit(1)

    df = pd.read_csv(RAW_PATH)
    print(f"Raw data loaded: {df.shape}")

    # fac_isWeekend is read as bool; convert to string for OHE
    if "fac_isWeekend" in df.columns and df["fac_isWeekend"].dtype == bool:
        df["fac_isWeekend"] = df["fac_isWeekend"].astype(str)

    fac_cols = [c for c in df.columns if c.startswith("fac_")]
    num_cols = [c for c in df.columns if c.startswith("num_")]

    for c in fac_cols:
        df[c] = df[c].fillna("UNKNOWN").astype(str)

    n_before = len(df)
    df = df.dropna(subset=num_cols)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} rows with NaN ({n_before} -> {len(df)})")
    else:
        print("No missing values found.")

    print(f"Features: {len(num_cols)} numeric, {len(fac_cols)} categorical")

    result = standard_pipeline(
        df,
        scaling=scaling,
        drop_first=drop_first,
        dropNZV=dropNZV,
        drop_high_corr=drop_high_corr,
        min_freq=min_freq,
        corr_threshold=corr_threshold,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)

    final_feats = [c for c in result.columns if c not in ("time", "event")]
    print(f"\nSaved to: {OUT_PATH}")
    print(f"Final: {result.shape[0]} rows, {len(final_feats)} features")
