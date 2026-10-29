"""
Preprocessing script for the METABRIC dataset.
-> metabric_genes : clinical features + PAM50 gene expression

Custom Preprocessing includes:
  - Binary encoding for ER / PR / HER2 status
  - Median imputation for missing values
  - StandardScaler for continuous features
  - Grade normalization: (grade - 1) / 2

Usage:
  python run_preprocess_metabric.py <method_name>
"""
import pandas as pd
import sys
import yaml
import os
from sklearn.impute import SimpleImputer
from src.utils_preprocessing import standard_pipeline


CLINICAL_FEATURES = {
    'Age at Diagnosis':              'num_age_at_diagnosis',
    'Tumor Size':                    'num_tumor_size',
    'Lymph nodes examined positive': 'num_lymph_nodes_positive',
    'Neoplasm Histologic Grade':     'num_neoplasm_histologic_grade',
    'ER Status':                     'fac_er_status',
    'PR Status':                     'fac_pr_status',
    'HER2 Status':                   'fac_her2_status',
}

OUTCOME_RENAME = {
    'Overall Survival (Months)': 'time',
    'Overall Survival Status':   'event',
}

BINARY_COLS = ['fac_er_status', 'fac_pr_status', 'fac_her2_status']
BINARY_MAP  = {'Negative': 0, 'Positive': 1}
EVENT_MAP   = {'0:LIVING': 0, '1:DECEASED': 1}

GRADE_COL = 'num_neoplasm_histologic_grade'


def load_and_prepare(config):
    clinical_df = pd.read_csv(
        config['source']['clinical_file'], delimiter='\t'
    )

    gene_df = pd.read_csv(
        config['source']['gene_expression_file'], delimiter='\t'
    )
    gene_names = sorted(gene_df.columns[2:])

    df = pd.merge(
        clinical_df, gene_df,
        left_on='Sample ID', right_on='SAMPLE_ID', how='inner',
    )

    rename_map = {**CLINICAL_FEATURES, **OUTCOME_RENAME}
    rename_map.update({g: f'num_{g}' for g in gene_names})
    cols_needed = list(CLINICAL_FEATURES.keys()) + list(OUTCOME_RENAME.keys()) + gene_names

    available = [c for c in cols_needed if c in df.columns]
    df = df[available].copy()
    df = df.rename(columns=rename_map)
    df = df.drop(columns=['Patient ID', 'Sample ID', 'SAMPLE_ID', 'STUDY_ID'],
                 errors='ignore')

    return df, gene_names


# Preprocessing
def prepare_for_pipeline(df, gene_names):
    """ METABRIC-specific proc before the shared standard_pipeline()."""
    time_col  = 'time'
    event_col = 'event'

    # Encode event status
    df[event_col] = df[event_col].map(EVENT_MAP)

    # Encode binary features to 0/1 and rename fac_ -> num_
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(BINARY_MAP)
            df = df.rename(columns={col: col.replace('fac_', 'num_')})

    # Drop missing outcomes, filter time > 0
    df = df.dropna(subset=[time_col, event_col])
    df = df[df[time_col] > 0].copy()
    print(f"Samples after outcome filtering: {len(df)}")

    # Drop rows where any gene expression value is NaN
    gene_cols = [f'num_{g}' for g in gene_names]
    existing_gene_cols = [c for c in gene_cols if c in df.columns]
    n_before = len(df)
    df = df.dropna(subset=existing_gene_cols)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} rows with missing gene expression values")

    # Median imputation for remaining missing numeric clinical features
    num_feature_cols = [c for c in df.columns
                        if c.startswith('num_') and c not in (time_col, event_col)]
    n_missing = int(df[num_feature_cols].isna().sum().sum())
    if n_missing > 0:
        imputer = SimpleImputer(strategy='median')
        df[num_feature_cols] = pd.DataFrame(
            imputer.fit_transform(df[num_feature_cols]),
            columns=num_feature_cols, index=df.index,
        )
        print(f"Imputed {n_missing} missing values (median strategy)")

    # Grade normalization: maps grades 1/2/3 -> 0/0.5/1
    if GRADE_COL in df.columns:
        df[GRADE_COL] = (df[GRADE_COL] - 1) / 2.0

    return df


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_preprocess_metabric.py <method_name>")
        print("  method_name: name of method config in ../Model_Training/method_configs/")
        sys.exit(1)

    method_name  = sys.argv[1]
    dataset_name = 'metabric_genes'

    #load dataset config
    config_path = f'configs/{dataset_name}.yaml'
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    #load method config
    method_config_path = f'../Model_Training/method_configs/{method_name}.yaml'
    if not os.path.exists(method_config_path):
        print(f"Error: Method config '{method_config_path}' not found")
        sys.exit(1)

    with open(method_config_path) as f:
        method_config = yaml.safe_load(f)

    pre            = method_config.get('preprocessing', {})
    scaling        = pre.get('scaling', True)
    drop_first     = pre.get('drop_first', True)
    dropNZV        = pre.get('drop_nzv', True)
    drop_high_corr = pre.get('drop_high_corr', True)
    min_freq       = pre.get('min_freq', 0.0025)
    corr_threshold = pre.get('corr_threshold', 0.98)

    print("=" * 60)
    print(f"METABRIC Preprocessing: {dataset_name}")
    print(f"\tMethod config: {method_name}")
    print(f"\tScaling: {scaling}")
    print(f"\tDrop first OHE cat: {drop_first}")
    print(f"\tDrop NZV: {dropNZV} (min_freq={min_freq})")
    print(f"\tDrop high correlation: {drop_high_corr} (threshold={corr_threshold})")
    print("=" * 60)

    df, gene_names = load_and_prepare(config)
    print(f"Raw data shape: {df.shape}")

    df = prepare_for_pipeline(df, gene_names)

    # share pipeline 
    dataset = standard_pipeline(
        df,
        scaling=scaling,
        drop_first=drop_first,
        dropNZV=dropNZV,
        drop_high_corr=drop_high_corr,
        min_freq=min_freq,
        corr_threshold=corr_threshold,
        exclude_from_scaling=[GRADE_COL],
    )

    output_dir  = f'data/processed/{dataset_name}'
    output_path = f'{output_dir}/{dataset_name}_processed.csv'
    os.makedirs(output_dir, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
