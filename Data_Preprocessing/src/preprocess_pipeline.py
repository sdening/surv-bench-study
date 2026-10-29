"""
Preprocessing pipeline called from modeling scripts.
"""
import pandas as pd
import yaml
import os
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector, ColumnTransformer
#from sklearn.impute import SimpleImputer
from sksurv.util import Surv
from SurvSet.data import SurvLoader
from src.utils_preprocessing import *


def run_preprocessing(dataset_name: str, method_name: str = None, output_dir: Path = None):
    """
    Run preprocessing for a dataset with method-specific parameters.
    
    Args:
        dataset_name: Name of the dataset to preprocess
        method_name: Name of the method (e.g., 'coxph'). 
        output_dir: Optional output directory. If None, uses data/processed/{dataset_name}/{method_name}/
        
    Returns:
        Path to the saved processed CSV file
    """
    # Get paths
    preprocess_root = Path(__file__).resolve().parent.parent
    
    # Load dataset config
    dataset_config_path = preprocess_root / 'configs' / f'{dataset_name}.yaml'
    with open(dataset_config_path, 'r') as f:
        content = f.read()
        config = yaml.safe_load(content)
        
        # Parse = syntax lines
        for line in content.split('\n'):
            if '=' in line and not line.strip().startswith('#'):
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"')
                config[key] = value
    
    # Load method-specific preprocessing settings if method_name provided
    if method_name:
        method_config_path = preprocess_root.parent / 'Model_Training' / 'method_configs' / f'{method_name}.yaml'
        with open(method_config_path, 'r') as f:
            method_config = yaml.safe_load(f)
        preproc_config = method_config.get('preprocessing', {})
        
        dropNZV = preproc_config.get('drop_nzv', True)
        drop_high_corr = preproc_config.get('drop_high_corr', True)
        drop_first = preproc_config.get('drop_first', True)
        scaling = preproc_config.get('scaling', True)
        min_freq = preproc_config.get('min_freq', 0.0025)
        corr_threshold = preproc_config.get('corr_threshold', 0.98)
    else:
        # Default settings if no method specified
        print("No method specified. Using default preprocessing settings.")
        dropNZV = True
        drop_high_corr = True
        drop_first = True
        scaling = True
        min_freq = 0.0025
        corr_threshold = 0.98
    
    drop_method = "first"
    
    print("-" * 50)
    print("Preprocessing Settings:")
    print(f"\tDataset: {dataset_name}")
    print(f"\tMethod: {method_name if method_name else 'default'}")
    print(f"\tApply scaling: {scaling}")
    print(f"\tDrop first category in one-hot encoding: {drop_first}")
    print(f"\tDrop near-zero variance features: {dropNZV} with minimum frequency threshold: {min_freq}")
    print(f"\tDrop highly correlated features: {drop_high_corr} with correlation threshold: {corr_threshold}")
    print("-" * 50)
    
    # Set up pipeline
    enc_fac = Pipeline(steps=[('ohe', OneHotEncoder(drop=drop_method, sparse_output=False, handle_unknown='ignore'))])
    sel_fac = make_column_selector(pattern=f'^{config["CAT_PREFIX"]}')
    
    enc_num = Pipeline(steps=[('scale', StandardScaler())])
    sel_num = make_column_selector(pattern=f'^{config["NUM_PREFIX"]}')
    enc_df = ColumnTransformer(transformers=[('ohe', enc_fac, sel_fac), ('s', enc_num, sel_num)])
    enc_df.set_output(transform='pandas')
    
    # Validate dataset name matches config
    config_ds_name = config['source']['params']['ds_name']
    if config_ds_name != dataset_name:
        raise ValueError(f"Dataset name mismatch: config specifies '{config_ds_name}' but function called with '{dataset_name}'")
    
    # Load and preprocess dataset
    loader = SurvLoader()
    df, ref = loader.load_dataset(ds_name=config_ds_name).values()
    
    # Remove rows with non-pos time values
    dataset = df[df[config['TIME_COL']] > 0].copy()
    dataset = summarize_categories(dataset, prefix='fac_', max_categories=10)
    
    # Fit and transform
    senc = Surv()
    So = senc.from_arrays(dataset[config['EVENT_COL']].astype(bool), dataset[config['TIME_COL']])
    enc_df.fit(dataset)
    df_features = enc_df.transform(dataset)
    
    # Drop rows with NaN after encoding
    df_features = df_features.dropna()
    
    # Remove columns with near-zero variance
    if dropNZV:
        cols_to_drop_nzv = drop_nzv_with_prefixes(df_features, min_freq=min_freq)
    else:
        cols_to_drop_nzv = []
    df_features = df_features.drop(columns=cols_to_drop_nzv, errors='ignore')
    
    # Remove columns with high correlation
    if drop_high_corr:
        cols_to_drop_hc = drop_corr_cols(df_features, threshold=corr_threshold)
    else:
        cols_to_drop_hc = []
    df_features = df_features.drop(columns=cols_to_drop_hc, errors='ignore')
    
    # Final dataset - align indices after dropna
    dataset = pd.concat(objs=[dataset[[config['EVENT_COL'], config['TIME_COL']]].loc[df_features.index], df_features], axis=1)
    print("Final shape:", dataset.shape)
    
    # Save to method-specific folder
    if output_dir is None:
        if method_name:
            output_dir = preprocess_root / 'data' / 'processed' / dataset_name 
        #else:
        #    output_dir = preprocess_root / 'data' / 'processed' / dataset_name / 'default'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = output_dir / f'{dataset_name}_processed.csv'
    dataset.to_csv(output_path, index=False)
    print(f"Saved processed data to: {output_path}")
    
    return output_path
