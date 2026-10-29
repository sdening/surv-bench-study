"""
Preprocessing script for the KKBox churn-prediction survival dataset.

Source preprocessing follows mostly the notebook code:
https://github.com/georgehc/survival-kernets/blob/main/datasets.py:

Usage:
  python run_preprocess_kkbox.py <method_name>
"""
import pandas as pd
import numpy as np
import sys
import yaml
import os
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from pycox.datasets import kkbox
from src.utils_preprocessing import standard_pipeline


DATASET_NAME = "kkbox_100k"
CONFIG_PATH  = f"configs/{DATASET_NAME}.yaml"

NUMERICAL_FEATURES = [
    "log_days_between_subs",
    "n_prev_churns",
    "log_days_since_reg_init",
    "log_payment_plan_days",
    "log_plan_list_price",
    "log_actual_amount_paid",
    "age_at_start",
]
CATEGORICAL_FEATURES = [
    "gender",
    "city",
    "registered_via",
]
BINARY_FEATURES = [
    "no_prev_churns",
    "is_cancel",
    "strange_age",
    "is_auto_renew",
    "nan_days_since_reg_init",
]


def load_raw():
    """Load raw KKBox data from pycox and return a tidy DataFrame with prefixed columns."""
    df = kkbox.read_df()
    if df is None:
        print("KKBox data not found locally, downloading...")
        kkbox.download_kkbox()
        df = kkbox.read_df()
    print(f"Raw pycox shape: {df.shape}")

    out = pd.DataFrame(index=df.index)
    for col in NUMERICAL_FEATURES:
        out[f"num_{col}"] = df[col].astype(float)
    for col in CATEGORICAL_FEATURES:
        out[f"fac_{col}"] = df[col]
    for col in BINARY_FEATURES:
        out[f"num_{col}"] = df[col].astype(int)
    out["time"]  = df["duration"].astype(float)
    out["event"] = df["event"].astype(int)
    return out


def prepare_for_pipeline(df):
    df = df.dropna(subset=["time", "event"])
    df = df[df["time"] > 0].copy()
    print(f"Samples after outcome filtering: {len(df)}")

    if "fac_gender" in df.columns:
        df["fac_gender"] = df["fac_gender"].astype(object).fillna("n/a").astype(str)
    if "fac_city" in df.columns:
        s = df["fac_city"].astype(object).where(df["fac_city"].notna(), "0")
        df["fac_city"] = s.apply(lambda x: str(int(float(x))))
    if "fac_registered_via" in df.columns:
        s = df["fac_registered_via"].astype(object).where(df["fac_registered_via"].notna(), "0")
        df["fac_registered_via"] = s.apply(lambda x: str(int(float(x))))

    num_cols = [c for c in df.columns if c.startswith("num_")]
    n_missing = int(df[num_cols].isna().sum().sum())
    if n_missing > 0:
        imputer = SimpleImputer(strategy="median")
        df[num_cols] = pd.DataFrame(
            imputer.fit_transform(df[num_cols]),
            columns=num_cols, index=df.index,
        )
        print(f"Imputed {n_missing} missing values (median strategy)")

    return df


def subsample(df, sample_n, random_state=42):
    """Stratified subsample of sample_n rows, preserving event rate."""
    if sample_n >= len(df):
        print(f"sample_n={sample_n} >= dataset size ({len(df)}), skipping subsampling.")
        return df
    _, df_sample = train_test_split(
        df, test_size=sample_n, stratify=df["event"], random_state=random_state
    )
    df_sample = df_sample.reset_index(drop=True)
    print(f"Subsampled {len(df)} -> {len(df_sample)} rows (stratified by event, seed={random_state})")
    return df_sample


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_preprocess_kkbox.py <method_name>")
        print("method_name: name of method config in ../Model_Training/method_configs/")
        sys.exit(1)

    method_name = sys.argv[1]

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    method_config_path = f"../Model_Training/method_configs/{method_name}.yaml"
    if not os.path.exists(method_config_path):
        print(f"Error: Method config '{method_config_path}' not found")
        sys.exit(1)

    with open(method_config_path) as f:
        method_config = yaml.safe_load(f)

    sample_n       = config["source"].get("sample_n")
    pre            = method_config.get("preprocessing", {})
    scaling        = pre.get("scaling", True)
    drop_first     = pre.get("drop_first", True)
    dropNZV        = pre.get("drop_nzv", True)
    drop_high_corr = pre.get("drop_high_corr", True)
    min_freq       = pre.get("min_freq", 0.0025)
    corr_threshold = pre.get("corr_threshold", 0.98)

    print("=" * 60)
    print(f"KKBox Preprocessing  ({DATASET_NAME})")
    print(f"\tMethod config: {method_name}")
    print(f"\tSubsample n: {sample_n or 'none (full dataset)'}")
    print(f"\tScaling: {scaling}")
    print(f"\tDrop first OHE cat: {drop_first}")
    print(f"\tDrop NZV: {dropNZV} (min_freq={min_freq})")
    print(f"\tDrop high correlation: {drop_high_corr} (threshold={corr_threshold})")
    print("=" * 60)

    df = load_raw()
    df = prepare_for_pipeline(df)

    if sample_n:
        df = subsample(df, sample_n)

    dataset = standard_pipeline(
        df,
        scaling=scaling,
        drop_first=drop_first,
        dropNZV=dropNZV,
        drop_high_corr=drop_high_corr,
        min_freq=min_freq,
        corr_threshold=corr_threshold,
    )

    output_dir  = f"data/processed/{DATASET_NAME}"
    output_path = f"{output_dir}/{DATASET_NAME}_processed.csv"
    os.makedirs(output_dir, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
