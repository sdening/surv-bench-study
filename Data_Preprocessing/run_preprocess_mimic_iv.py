"""
Preprocessing script and information for the MIMIC-IV dataset.
References for preprocessing: 
    - https://arxiv.org/abs/2008.08637
    - and https://www.sciencedirect.com/science/article/pii/S1532046418300716

Source data:
    - https://physionet.org/content/mimiciv/3.1/
(all references see configs/mimic_iv.yaml)

Preprocessing steps:
Cohort:
  - Adult (>=18) first ICU stay per patient
  - Minimum 24h observation window (time >= 24h)

Survival framing:
  - event = 1 if deathtime is recorded (in-hospital mortality)
  - time  = (deathtime or dischtime) - ICU intime, in hours

Features (first-24h aggregation, mean):
  - Demographics (categorical): gender, race, insurance, marital_status,
                                admission_type, first_careunit
  - Chartevents vitals: HR, RR, Temp, SpO2, SBP, MBP, DBP
  - Labevents: creatinine, glucose, hematocrit, hemoglobin, platelets,
               WBC, potassium, sodium, bicarbonate, BUN, bilirubin, lactate

Missing handling:
  - Features with >50% missing are dropped
  - Remaining rows with any NaN are dropped 

Time handling:
  - Administrative censoring at 30 days (720h): events beyond this
    horizon are censored (event=0, time=720)

Usage:
  python run_preprocess_mimic_iv.py <method_name>
"""
import pandas as pd
import numpy as np
import sys
import yaml
import os
from pathlib import Path
from collections import defaultdict
from src.utils_preprocessing import standard_pipeline

#matching
CHART_FEATURE_ITEMIDS = {
    'heart_rate':   [220045],
    'resp_rate':    [220210],
    'temperature':  [223762],
    'spo2':         [220277],
    'sbp':          [220179],
    'mbp':          [220181],
    'dbp':          [220180],
}

LAB_FEATURE_ITEMIDS = {
    'creatinine':   [50912, 52546],
    'glucose_lab':  [50931, 50809, 52027],
    'hematocrit':   [50810, 52028],
    'hemoglobin':   [50811],
    'platelets':    [51704],
    'wbc':          [51755, 51756],
    'potassium':    [50971, 50822],
    'sodium':       [50983, 50824],
    'bicarbonate':  [50882],
    'bun':          [51006],
    'bilirubin':    [50885],
    'lactate':      [52442, 53154],
}

MISSING_THRESHOLD = 0.50
HORIZON_HOURS = 720  #cap

DEMOGRAPHIC_CAT = ['gender', 'race', 'insurance', 'marital_status',
                   'admission_type', 'admission_location', 'language',
                   'first_careunit']

# Cohort construction
def build_cohort(raw_dir):
    """Build survival-analysis cohort: first ICU stay per adult patient."""
    hosp = Path(raw_dir) / 'hosp'
    icu  = Path(raw_dir) / 'icu'

    patients = pd.read_csv(hosp / 'patients.csv.gz', compression='gzip')
    admissions = pd.read_csv(hosp / 'admissions.csv.gz', compression='gzip')
    icustays = pd.read_csv(icu / 'icustays.csv.gz', compression='gzip')

    for col in ['dod']:
        if col in patients.columns:
            patients[col] = pd.to_datetime(patients[col], errors='coerce')
    for col in ['admittime', 'dischtime', 'deathtime']:
        if col in admissions.columns:
            admissions[col] = pd.to_datetime(admissions[col], errors='coerce')
    for col in ['intime', 'outtime']:
        if col in icustays.columns:
            icustays[col] = pd.to_datetime(icustays[col], errors='coerce')

    cohort = icustays.merge(
        admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime',
                     'deathtime', 'hospital_expire_flag',
                     'admission_type', 'admission_location',
                     'insurance', 'language', 'marital_status', 'race']],
        on=['subject_id', 'hadm_id'], how='left',
    ).merge(
        patients[['subject_id', 'gender', 'anchor_age', 'dod']],
        on='subject_id', how='left',
    )

    cohort = cohort[cohort['anchor_age'] >= 18].copy()

    # Keep first ICU stay per patient 
    cohort = cohort.sort_values(['subject_id', 'intime'])
    cohort = cohort.drop_duplicates(subset='subject_id', keep='first')

    cohort['event'] = cohort['deathtime'].notna().astype(int)
    death_or_discharge = cohort['deathtime'].fillna(cohort['dischtime'])
    cohort['time'] = (death_or_discharge - cohort['intime']).dt.total_seconds() / 3600.0

    cohort = cohort[
        cohort['time'].notna() &
        (cohort['time'] > 0) &
        cohort['event'].notna() &
        cohort['intime'].notna() &
        cohort['outtime'].notna()
    ].copy()

    cohort = cohort[cohort['time'] >= 24].copy()

    # events beyond horizon are censored
    cohort.loc[cohort['time'] > HORIZON_HOURS, 'event'] = 0
    cohort.loc[cohort['time'] > HORIZON_HOURS, 'time'] = HORIZON_HOURS

    for col in DEMOGRAPHIC_CAT:
        if col in cohort.columns:
            cohort[col] = cohort[col].fillna('UNKNOWN').astype(str)

    cohort = cohort[[
        'subject_id', 'hadm_id', 'stay_id',
        'anchor_age', 'admittime', 'dischtime', 'deathtime',
        'intime', 'outtime', 'time', 'event',
    ] + [c for c in DEMOGRAPHIC_CAT if c in cohort.columns]].copy()

    print(f"Cohort: {len(cohort)} patients, "
          f"{cohort['event'].sum()} events ({cohort['event'].mean()*100:.1f}%)")
    return cohort



# First-24h vitals values aggregation
def aggregate_chartevents_first24h(raw_dir, cohort_stays, chunksize=2_000_000):
    """Aggregate vital-sign chartevents from first 24h of ICU stay (mean)."""
    chartevents_fp = Path(raw_dir) / 'icu' / 'chartevents.csv.gz'

    itemid_to_feature = {}
    for feat, ids in CHART_FEATURE_ITEMIDS.items():
        for iid in ids:
            itemid_to_feature[iid] = feat

    selected_itemids = set(itemid_to_feature.keys())
    stay_info = cohort_stays.set_index('stay_id')[['intime', 'window_end']]
    target_stays = set(cohort_stays['stay_id'].tolist())

    agg_sum = defaultdict(float)
    agg_count = defaultdict(int)

    for i, chunk in enumerate(pd.read_csv(
        chartevents_fp, compression='gzip',
        usecols=['stay_id', 'charttime', 'itemid', 'valuenum'],
        chunksize=chunksize,
    )):
        chunk = chunk[
            chunk['stay_id'].isin(target_stays) &
            chunk['itemid'].isin(selected_itemids) &
            chunk['valuenum'].notna()
        ].copy()
        if chunk.empty:
            continue

        chunk['charttime'] = pd.to_datetime(chunk['charttime'], errors='coerce')
        chunk = chunk.dropna(subset=['charttime'])
        chunk = chunk.join(stay_info, on='stay_id', how='left')
        chunk = chunk[
            (chunk['charttime'] >= chunk['intime']) &
            (chunk['charttime'] <= chunk['window_end'])
        ]
        if chunk.empty:
            continue

        chunk['feature'] = chunk['itemid'].map(itemid_to_feature)
        grouped = chunk.groupby(['stay_id', 'feature'])['valuenum'].agg(['sum', 'count']).reset_index()

        for _, row in grouped.iterrows():
            key = (row['stay_id'], row['feature'])
            agg_sum[key] += row['sum']
            agg_count[key] += row['count']

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1} chartevents chunks...")

    rows = []
    for (stay_id, feature), s in agg_sum.items():
        c = agg_count[(stay_id, feature)]
        rows.append({'stay_id': stay_id, 'feature': feature, 'mean_value': s / c})

    if not rows:
        return pd.DataFrame(columns=['stay_id'])

    long_df = pd.DataFrame(rows)
    wide_df = long_df.pivot(index='stay_id', columns='feature', values='mean_value').reset_index()
    wide_df.columns.name = None
    return wide_df

# First-24h labevents aggregation
def aggregate_labevents_first24h(raw_dir, cohort_stays, chunksize=1_000_000):
    """Aggregate lab values from first 24h of ICU stay (mean)."""
    labevents_fp = Path(raw_dir) / 'hosp' / 'labevents.csv.gz'

    itemid_to_feature = {}
    for feat, ids in LAB_FEATURE_ITEMIDS.items():
        for iid in ids:
            itemid_to_feature[iid] = feat

    selected_itemids = set(itemid_to_feature.keys())
    target_hadm_ids = set(cohort_stays['hadm_id'].dropna().tolist())
    hadm_info = cohort_stays.set_index('hadm_id')[['intime', 'window_end']]

    agg_sum = defaultdict(float)
    agg_count = defaultdict(int)

    for i, chunk in enumerate(pd.read_csv(
        labevents_fp, compression='gzip',
        usecols=['hadm_id', 'charttime', 'itemid', 'valuenum'],
        chunksize=chunksize,
    )):
        chunk = chunk[
            chunk['hadm_id'].isin(target_hadm_ids) &
            chunk['itemid'].isin(selected_itemids) &
            chunk['valuenum'].notna()
        ].copy()
        if chunk.empty:
            continue

        chunk['charttime'] = pd.to_datetime(chunk['charttime'], errors='coerce')
        chunk = chunk.dropna(subset=['charttime'])
        chunk = chunk.join(hadm_info, on='hadm_id', how='left')
        chunk = chunk[
            (chunk['charttime'] >= chunk['intime']) &
            (chunk['charttime'] <= chunk['window_end'])
        ]
        if chunk.empty:
            continue

        chunk['feature'] = chunk['itemid'].map(itemid_to_feature)
        grouped = chunk.groupby(['hadm_id', 'feature'])['valuenum'].agg(['sum', 'count']).reset_index()

        for _, row in grouped.iterrows():
            key = (row['hadm_id'], row['feature'])
            agg_sum[key] += row['sum']
            agg_count[key] += row['count']

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1} labevents chunks...")

    rows = []
    for (hadm_id, feature), s in agg_sum.items():
        c = agg_count[(hadm_id, feature)]
        rows.append({'hadm_id': hadm_id, 'feature': feature, 'mean_value': s / c})

    if not rows:
        return pd.DataFrame(columns=['hadm_id'])

    long_df = pd.DataFrame(rows)
    wide_df = long_df.pivot(index='hadm_id', columns='feature', values='mean_value').reset_index()
    wide_df.columns.name = None

    stay_lookup = cohort_stays[['hadm_id', 'stay_id']].drop_duplicates()
    wide_df = wide_df.merge(stay_lookup, on='hadm_id', how='left')
    wide_df = wide_df.drop(columns=['hadm_id'])
    return wide_df


# Prepare for standard pipeline
def prepare_for_pipeline(df):
    meta_cols = ['subject_id', 'hadm_id', 'stay_id', 'anchor_age',
                 'admittime', 'dischtime', 'deathtime', 'intime', 'outtime']
    outcome_cols = ['time', 'event']
    feature_cols = [c for c in df.columns if c not in meta_cols + outcome_cols]

    num_feature_cols = [c for c in feature_cols if c not in DEMOGRAPHIC_CAT]
    missing_rates = df[num_feature_cols].isna().mean().sort_values(ascending=False)
    high_missing = missing_rates[missing_rates > MISSING_THRESHOLD].index.tolist()
    if high_missing:
        print(f"Dropping {len(high_missing)} features with >{MISSING_THRESHOLD*100:.0f}% missing: "
              f"{high_missing}")
    keep_num = [c for c in num_feature_cols if c not in high_missing]
    keep_cat = [c for c in DEMOGRAPHIC_CAT if c in df.columns]

    result = df[outcome_cols + keep_num + keep_cat].copy()

    n_before = len(result)
    result = result.dropna(subset=keep_num)
    n_dropped = n_before - len(result)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} rows with missing values "
              f"({len(result)} remaining)")

    rename = {}
    for c in keep_num:
        rename[c] = f'num_{c}'
    for c in keep_cat:
        rename[c] = f'fac_{c}'
    result = result.rename(columns=rename)
    result = result.reset_index(drop=True)

    print(f"Numeric features: {len(keep_num)}, Categorical features: {len(keep_cat)}")
    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_preprocess_mimic_iv.py <method_name>")
        print("  method_name: name of method config in ../Model_Training/method_configs/")
        sys.exit(1)

    dataset_name = 'mimic_iv'
    method_name  = sys.argv[1]

    config_path = 'configs/mimic_iv.yaml'
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    method_config_path = f'../Model_Training/method_configs/{method_name}.yaml'
    if not os.path.exists(method_config_path):
        print(f"Error: Method config '{method_config_path}' not found")
        sys.exit(1)

    with open(method_config_path, 'r') as f:
        method_config = yaml.safe_load(f)

    pre = method_config.get('preprocessing', {})
    scaling        = pre.get('scaling', True)
    drop_first     = pre.get('drop_first', True)
    dropNZV        = pre.get('drop_nzv', True)
    drop_high_corr = pre.get('drop_high_corr', True)
    min_freq       = pre.get('min_freq', 0.0025)
    corr_threshold = pre.get('corr_threshold', 0.98)

    raw_dir = config['source']['raw_dir']

    print("=" * 60)
    print(f"MIMIC-IV Preprocessing: {dataset_name}")
    print(f"\tMethod config: {method_name}")
    print(f"\tRaw data dir: {raw_dir}")
    print(f"\tTime horizon: {HORIZON_HOURS}h ({HORIZON_HOURS/24:.0f} days)")
    print(f"\tMissing threshold: {MISSING_THRESHOLD}")
    print(f"\tScaling: {scaling}")
    print(f"\tDrop first OHE cat: {drop_first}")
    print(f"\tDrop NZV: {dropNZV} (min_freq={min_freq})")
    print(f"\tDrop high correlation: {drop_high_corr} (threshold={corr_threshold})")
    print("=" * 60)

    #Build cohort
    print("\n[1/3] Building cohort...")
    cohort = build_cohort(raw_dir)
    #Prepare 24h windows
    cohort_stays = cohort[['subject_id', 'hadm_id', 'stay_id', 'intime']].copy()
    cohort_stays['window_end'] = cohort_stays['intime'] + pd.Timedelta(hours=24)

    # Aggregate
    print("\n[2/3] Aggregating first-24h...")
    chart_wide = aggregate_chartevents_first24h(raw_dir, cohort_stays)
    lab_wide = aggregate_labevents_first24h(raw_dir, cohort_stays)

    final_df = cohort.copy()
    if 'stay_id' in chart_wide.columns:
        final_df = final_df.merge(chart_wide, on='stay_id', how='left')
    if 'stay_id' in lab_wide.columns:
        final_df = final_df.merge(lab_wide, on='stay_id', how='left')

    final_df['age'] = final_df['anchor_age']
    print(f"\nMerged dataset shape: {final_df.shape}")

    print("\n[3/3] Preparing for standard pipeline...")
    df = prepare_for_pipeline(final_df)
    print(f"Pre-pipeline shape: {df.shape}")

    dataset = standard_pipeline(
        df,
        scaling=scaling,
        drop_first=drop_first,
        dropNZV=dropNZV,
        drop_high_corr=drop_high_corr,
        min_freq=min_freq,
        corr_threshold=corr_threshold,
    )

    output_dir = f'data/processed/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/{dataset_name}_processed.csv'
    dataset.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"Final shape: {dataset.shape}")
