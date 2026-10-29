import pandas as pd
import numpy as np
import sys
import yaml
import os
from sklearn.impute import SimpleImputer
from src.utils_preprocessing import standard_pipeline


DATASET_NAME = "ADNI_dem_biom_hipp"
CONFIG_PATH  = f"configs/{DATASET_NAME}.yaml"

NUM_FEATURES = [
    "AGE", "PTEDUCAT",
    "CSF", "FDG", "AV45", "ABETA", "TAU", "PTAU",
    "SurfaceHoles", "TotalGrayVol", "Optic-Chiasm", "5th-Ventricle",
    "Left-Hippocampus-normalized",
]
CAT_FEATURES = ["PTGENDER", "ATN_status"]
VOLUME_RAW_COLS = ["Left-Hippocampus", "eTIV"] #hippocampus normalized with eTIV
VOLUME_DERIVED  = "Left-Hippocampus-normalized"
OUTCOME_COLS = ["event", "time"]
ID_COL = "RID"


def load_raw(config):
    raw_path = config["source"].get("raw_path")
    if raw_path and os.path.exists(raw_path):
        print(f"Loading from raw source: {raw_path}")
        return pd.read_csv(raw_path)
    print(f"Error: Raw source '{raw_path}' not found.")
    print("Place the raw ADNI CSV at the path specified in the config.")
    sys.exit(1)


def select_and_rename(df):
    """Select required columns, normalize hippocampus volume, rename with prefixes."""
    needed = [ID_COL] + OUTCOME_COLS + NUM_FEATURES + CAT_FEATURES + VOLUME_RAW_COLS
    needed = [c for c in dict.fromkeys(needed) if c != VOLUME_DERIVED]

    available = [c for c in needed if c in df.columns]
    missing = set(needed) - set(available)
    if missing:
        print(f"Warning: columns not found in raw data: {missing}")
    df = df[available].copy()

    # Normalize left hippocampus volume by eTIV
    if "Left-Hippocampus" in df.columns and "eTIV" in df.columns:
        df[VOLUME_DERIVED] = df["Left-Hippocampus"] / df["eTIV"]
        df = df.drop(columns=["Left-Hippocampus", "eTIV"])
        print(f"Created {VOLUME_DERIVED} = Left-Hippocampus / eTIV")

    single_unique = [c for c in df.columns if df[c].dropna().nunique() == 1]
    if single_unique:
        print(f"Dropping {len(single_unique)} single-unique columns: {single_unique}")
        df = df.drop(columns=single_unique, errors="ignore")

    rename_dict = {c: f"fac_{c}" for c in CAT_FEATURES if c in df.columns}
    rename_dict.update({c: f"num_{c}" for c in NUM_FEATURES if c in df.columns})
    df = df.rename(columns=rename_dict)

    if "event" in df.columns and df["event"].dtype == object:
        df["event"] = df["event"].astype(str).str.lower().map({"yes": 1, "no": 0})

    return df


def prepare_for_pipeline(df):
    """ADNI-specific cleaning: event encoding, outcome filtering, median imputation."""
    if df["event"].dtype == object or df["event"].dtype == bool:
        df["event"] = df["event"].astype(str).str.lower().map({
            "true": 1, "false": 0,
            "yes": 1, "no": 0,
            "1": 1, "0": 0, "1.0": 1, "0.0": 0,
        })
    df["event"] = df["event"].astype(float)
    df = df.dropna(subset=["time", "event"])
    df = df[df["time"] > 0].copy()
    print(f"Samples after outcome filtering: {len(df)}")

    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_preprocess_ADNI.py <method_name>")
        print("  method_name: name of method config in ../Model_Training/method_configs/")
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

    pre            = method_config.get("preprocessing", {})
    scaling        = pre.get("scaling", True)
    drop_first     = pre.get("drop_first", True)
    dropNZV        = pre.get("drop_nzv", True)
    drop_high_corr = pre.get("drop_high_corr", True)
    min_freq       = pre.get("min_freq", 0.0025)
    corr_threshold = pre.get("corr_threshold", 0.98)

    print("=" * 60)
    print(f"ADNI Preprocessing: {DATASET_NAME}")
    print(f"\tMethod config: {method_name}")
    print(f"\tScaling: {scaling}")
    print(f"\tDrop first OHE cat: {drop_first}")
    print(f"\tDrop NZV: {dropNZV} (min_freq={min_freq})")
    print(f"\tDrop high correlation: {drop_high_corr} (threshold={corr_threshold})")
    print("=" * 60)

    df = load_raw(config)
    print(f"Raw data shape: {df.shape}")

    df = select_and_rename(df)
    print(f"After column selection: {df.shape}")

    df = prepare_for_pipeline(df)

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
