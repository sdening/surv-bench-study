"""
preprocessing script for all SA datasets (mainly from survset)
SurvSet package Drysdale et al. (2022)
"""
import pandas as pd
import sys
import yaml
import os
from SurvSet.data import SurvLoader
from src.utils_preprocessing import standard_pipeline

if len(sys.argv) < 3:
    print("Usage: python run_preprocess.py <dataset_name> <method_name>")
    sys.exit(1)
dataset_name = sys.argv[1]
method_name = sys.argv[2]

# Load dataset config
with open(f'configs/{dataset_name}.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load method preprocessing config
method_config_path = f'../Model_Training/method_configs/{method_name}.yaml'
with open(method_config_path, 'r') as f:
    method_config = yaml.safe_load(f)

pre = method_config.get('preprocessing', {})
scaling        = pre.get('scaling', True)
drop_first     = pre.get('drop_first', True)
dropNZV        = pre.get('drop_nzv', True)
drop_high_corr = pre.get('drop_high_corr', True)
min_freq       = pre.get('min_freq', 0.0025)
corr_threshold = pre.get('corr_threshold', 0.98)

print("-" * 50)
print("Preprocessing Settings:")
print(f"\tDataset: {dataset_name}")
print(f"\tMethod: {method_name}")
print(f"\tScaling: {scaling}")
print(f"\tDrop first OHE cat: {drop_first}")
print(f"\tDrop NZV: {dropNZV} (min_freq={min_freq})")
print(f"\tDrop high corr: {drop_high_corr} (threshold={corr_threshold})")
print("-" * 50)

DO_NOT_OVERWRITE = {"support2"}
if dataset_name in DO_NOT_OVERWRITE:
    print(f"ERROR: '{dataset_name}' uses external preprocessing and must not "
          f"be re-processed via this script. Exiting.")
    sys.exit(1)

# Dataset-specific: load from SurvSet
loader = SurvLoader()
df, ref = loader.load_dataset(
    ds_name=config['source']['params']['ds_name']
).values()

tc = config['TIME_COL']
ec = config['EVENT_COL']

dataset = df[df[tc] > 0].copy()

#  Shared  pipeline 
dataset = standard_pipeline(
    dataset,
    time_col=tc,
    event_col=ec,
    scaling=scaling,
    drop_first=drop_first,
    dropNZV=dropNZV,
    drop_high_corr=drop_high_corr,
    min_freq=min_freq,
    corr_threshold=corr_threshold,
)

# Save
os.makedirs(f'data/processed/{dataset_name}', exist_ok=True)
dataset.to_csv(
    f'data/processed/{dataset_name}/{dataset_name}_processed.csv', index=False
)
print(f"Saved to: data/processed/{dataset_name}/{dataset_name}_processed.csv")

