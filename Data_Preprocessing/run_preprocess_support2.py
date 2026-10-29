"""
Preprocessing script for the SUPPORT2 dataset.

Reproduces the preprocessing from:
(https://github.com/georgehc/survival-intro/blob/main/datasets.py),
which follows the DeepSurv paper (Katzman et al., 2018).

Usage:
  python run_preprocess_support2.py <method_name>
"""
import pandas as pd
import numpy as np
import os
import sys
import yaml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.utils_preprocessing import (
    summarize_categories, drop_nzv_with_prefixes, drop_corr_cols,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(SCRIPT_DIR, "data", "support2", "raw", "support2.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "data", "processed", "support2")
OUT_PATH = os.path.join(OUT_DIR, "support2_processed.csv")

FEATURES_SELECTED = [
    'age', 'sex', 'race', 'num.co', 'diabetes',
    'dementia', 'ca', 'meanbp', 'hrt', 'resp',
    'temp', 'wblc', 'sod', 'crea',
]

FEATURES_AFTER = [
    's__num_age', 'num_female', 'num_numco', 'num_diabetes', 'num_dementia',
    'num_ca', 's__num_meanbp', 's__num_hrt', 's__num_resp', 's__num_temp',
    's__num_wblc', 's__num_sod', 's__num_crea',
    'ohe__fac_race_blank', 'ohe__fac_race_asian', 'ohe__fac_race_black',
    'ohe__fac_race_hispanic', 'ohe__fac_race_other', 'ohe__fac_race_white',
]

COLS_STANDARDIZE = [0, 7, 8, 9, 10, 11, 12, 13]  # age, meanbp..crea
COLS_LEAVE = [1, 4, 5]                             # female, diabetes, dementia
COLS_CATEGORICAL = [2]                             # race


def _map_cancer(x): #helper
    if x == 'metastatic':
        return 2
    elif x == 'yes':
        return 1
    return 0


def load_and_clean(raw_path):
    """Load raw support2.csv and apply https://github.com/georgehc/survival-intro/blob/main/datasets.py cleaning steps."""
    raw_df = pd.read_csv(raw_path)
    print(f"Raw data loaded: {raw_df.shape}")

    df = raw_df[FEATURES_SELECTED + ['d.time', 'death']].copy()
    df = df.astype({'race': 'category'})

    df['sex'] = df['sex'].apply(lambda x: 1 * (x == 'female'))
    df = df.rename(columns={'sex': 'female'})
    df['ca'] = df['ca'].apply(_map_cancer)

    df['race'] = df['race'].cat.codes + 1

    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} rows with NaN ({n_before} -> {len(df)})")

    return df


def preprocess(df):
    """see https://github.com/georgehc/survival-intro/blob/main/datasets.py"""
    feature_cols = ['age', 'female', 'race', 'num.co', 'diabetes',
                    'dementia', 'ca', 'meanbp', 'hrt', 'resp',
                    'temp', 'wblc', 'sod', 'crea']

    X = df[feature_cols].to_numpy().astype('float32')
    Y = df['d.time'].to_numpy().astype('float32')
    D = df['death'].to_numpy().astype('int32')

    categories = [list(range(int(X[:, 2].max()) + 1))]

    new_features = np.zeros((X.shape[0], 19), dtype='float32')
    scaler = StandardScaler()
    encoder = OneHotEncoder(categories=categories)

    new_features[:, [0, 6, 7, 8, 9, 10, 11, 12]] = \
        scaler.fit_transform(X[:, COLS_STANDARDIZE])
    new_features[:, [1, 3, 4]] = X[:, COLS_LEAVE]
    new_features[:, 13:] = \
        encoder.fit_transform(X[:, COLS_CATEGORICAL]).toarray()
    new_features[:, 2] = X[:, 3] / 9.0   # num.co / 9
    new_features[:, 5] = X[:, 6] / 2.0   # ca / 2

    result = pd.DataFrame(new_features, columns=FEATURES_AFTER)
    result['time'] = Y
    result['event'] = D

    return result


def main(method_name):
    method_config_path = os.path.join(
        SCRIPT_DIR, "..", "Model_Training", "method_configs", f"{method_name}.yaml"
    )
    with open(method_config_path, "r") as f:
        method_config = yaml.safe_load(f)

    dropNZV       = method_config.get("dropNZV", True)
    drop_high_corr = method_config.get("drop_high_corr", True)
    corr_threshold = method_config.get("corr_threshold", 0.98)

    print("-" * 50)
    print(f"\tDataset:\t\tsupport2")
    print(f"\tMethod:\t\t\t{method_name}")
    print("-" * 50)

    if not os.path.exists(RAW_PATH):
        print(f"ERROR: Raw data not found at {RAW_PATH}")
        sys.exit(1)

    df = load_and_clean(RAW_PATH)

    result = preprocess(df)
    print(f"After reference preprocessing: {result.shape}")

    feat_cols = [c for c in result.columns if c not in ('time', 'event')]
    feat_df = result[feat_cols].copy()

    result = summarize_categories(result, prefix='fac_', max_categories=10)

    if dropNZV:
        nzv_drop = drop_nzv_with_prefixes(feat_df)
        if nzv_drop:
            result = result.drop(columns=nzv_drop, errors='ignore')

    if drop_high_corr:
        corr_drop = drop_corr_cols(feat_df, threshold=corr_threshold)
        if corr_drop:
            result = result.drop(columns=corr_drop, errors='ignore')

    os.makedirs(OUT_DIR, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to: {OUT_PATH}")
    final_feats = [c for c in result.columns if c not in ('time', 'event')]
    print(f"Final: {result.shape[0]} subjects, {len(final_feats)} features")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python run_preprocess_support2.py <method_name>")
        sys.exit(1)
    main(sys.argv[1])
