#from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


## Helper Functions
def _is_binary_01(series: pd.Series) -> bool:
    """Return True if a numeric column contains only values in {0, 1}."""
    unique = series.dropna().unique()
    return len(unique) <= 2 and set(unique).issubset({0, 0.0, 1, 1.0})


## Main Functions

def summarize_categories(
        df: pd.DataFrame, 
        prefix: str = 'fac_', 
        max_categories: int = 10
    ):
   
   # Get categorical columns by dtype or prefix 
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols.extend([col for col in df.columns if col.startswith(prefix) and col not in cat_cols])
    
    for col in cat_cols:
        counts = df[col].value_counts()
        
        if len(counts) > max_categories: #if there are more than 10 categories -> summarze rest to "other"
            print(f"\nColumn {col} has {len(counts)} categories. Value counts:")
            print(counts)
            
            top_categories = counts.nlargest(max_categories).index
            df[col] = df[col].apply(lambda x: x if x in top_categories else 'other')
            # Convert all values to strings to ensure uniform data type
            df[col] = df[col].astype(str)
            
            print(f"\nAfter grouping to {max_categories} categories:")
            print(df[col].value_counts())
            
    return df

def drop_nzv_with_prefixes(
    X: pd.DataFrame,
    min_std_num: float = 1e-5,   
    min_freq: float = 0.0025,     
    min_count: int = 5           
    ):
    n = len(X)

    num_cols = [c for c in X.columns if c.startswith(("num_", "s__num_"))]
    ohe_cols = [c for c in X.columns if c.startswith("ohe__")]
    drop_num, drop_ohe = [], []

    #check variance of 'num_' cols 
    if num_cols:
        std = X[num_cols].std()
        drop_num = std[std < min_std_num].index.tolist()
        if drop_num:
            print(f"\nNum cols dropped (std < {min_std_num}):")
            for c in drop_num:
                print(f"  {c:<40} std={std[c]:.3e}")

    # check var of 'ohe__' cols (depending on the dataset size)
    if ohe_cols:
        p = X[ohe_cols].mean(axis=0) #p for prevelance            
        min_p = max(min_freq, min_count / max(n, 1)) #threshold dependent on dataset size
        var = p * (1 - p)
        var_thresh = min_p * (1 - min_p)
        drop_ohe = var[var < var_thresh].index.tolist()
        if drop_ohe:
            print(f"\nOHE cols dropped (prevelance < {min_p:.4f} or > {1-min_p:.4f}):")
            for c in drop_ohe:
                count_1 = int(X[c].sum())
                count_0 = n - count_1
                print(f"  {c:<40} prevelance={p[c]:.4f}, var={var[c]:.3e}, count_1={count_1}, count_0={count_0}, n={n}")
    else:
        min_p = max(min_freq, min_count / max(n, 1))

    drop = sorted(set(drop_num) | set(drop_ohe))
    
    print(f"Dropping {len(drop)} cols because of near zero variance: {drop}")
    return drop


def drop_corr_cols(
        X: pd.DataFrame, 
        threshold: float = 0.98
    ):
    method = "spearman"
    corr = X.corr(method = method).abs() #calculate  correlation
    mask_upper = np.triu(np.ones(corr.shape, dtype=bool), k=1) #upper triangle of corr matrix
    upper = corr.where(mask_upper)

    pairs = []
    for j, col in enumerate(upper.columns):
        hits = upper.index[upper[col] > threshold].tolist()
        for i_name in hits:
            pairs.append((i_name, col, float(upper.loc[i_name, col])))

    print(f"\nPairs with '{method}' > {threshold}:\n")
    for a, b, cval in sorted(pairs, key=lambda t: t[2], reverse=True):
        print(f"  {a:<32} <-> {b:<32} corr={cval:.3f}")

    drop = sorted(set(b for _, b, _ in pairs))

    print(f"Dropping {len(drop)} cols because of high correlation: {drop}")
    return drop



def standard_pipeline(
    df: pd.DataFrame,
    time_col: str = 'time',
    event_col: str = 'event',
    scaling: bool = True,
    drop_first: bool = True,
    dropNZV: bool = True,
    drop_high_corr: bool = True,
    min_freq: float = 0.0025,
    corr_threshold: float = 0.98,
    exclude_from_scaling: list = None,
):
    if exclude_from_scaling is None:
        exclude_from_scaling = []

    # Category summarization
    df = summarize_categories(df, prefix='fac_', max_categories=10)

    # Separate outcomes from features
    outcome_df = df[[event_col, time_col]].copy()
    num_cols = [c for c in df.columns if c.startswith('num_')]
    fac_cols = [c for c in df.columns if c.startswith('fac_')]
    feature_df = df[num_cols + fac_cols].copy()

    # OHE on fac_ columns
    drop_method = 'first' if drop_first else None
    try:
        ohe = OneHotEncoder(drop=drop_method, sparse_output=False,
                            handle_unknown='ignore')
    except TypeError:
        ohe = OneHotEncoder(drop=drop_method, sparse=False,
                            handle_unknown='ignore')

    if fac_cols:
        ohe_encoded = ohe.fit_transform(feature_df[fac_cols])
        raw_names = ohe.get_feature_names_out(fac_cols)
        ohe_col_names = [f'ohe__{n}' for n in raw_names]
        ohe_df = pd.DataFrame(ohe_encoded, columns=ohe_col_names,
                              index=feature_df.index)
        feature_df = feature_df.drop(columns=fac_cols)
        feature_df = pd.concat([feature_df, ohe_df], axis=1)

    # StandardScaler on num_ columns (exclude binary 0/1 and explicit list)
    cols_to_scale = []
    if scaling and num_cols:
        cols_to_scale = [
            c for c in num_cols
            if c not in exclude_from_scaling and not _is_binary_01(feature_df[c])
        ]
        if cols_to_scale:
            scaler = StandardScaler()
            feature_df[cols_to_scale] = scaler.fit_transform(
                feature_df[cols_to_scale]
            )
            skipped = set(num_cols) - set(cols_to_scale)
            if skipped:
                print(f"Scaling {len(cols_to_scale)} num_ cols, "
                      f"skipped {len(skipped)} (binary/excluded): "
                      f"{sorted(skipped)}")

    # Rename scaled columns: num_ -> s__num_ (matches ColumnTransformer convention)
    if cols_to_scale:
        scale_rename = {c: f's__{c}' for c in cols_to_scale
                        if c in feature_df.columns}
        feature_df = feature_df.rename(columns=scale_rename)

    # Drop NaN rows that may remain after OHE
    initial_rows = len(feature_df)
    feature_df = feature_df.dropna()
    dropped_rows = initial_rows - len(feature_df)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with missing values "
              f"({dropped_rows / initial_rows * 100:.2f}%)")
        outcome_df = outcome_df.loc[feature_df.index]

    # NZV removal
    if dropNZV:
        cols_to_drop = drop_nzv_with_prefixes(feature_df, min_freq=min_freq)
        feature_df = feature_df.drop(columns=cols_to_drop, errors='ignore')

    # High correlation removal
    if drop_high_corr:
        cols_to_drop = drop_corr_cols(feature_df, threshold=corr_threshold)
        feature_df = feature_df.drop(columns=cols_to_drop, errors='ignore')

    # Reassemble with this spec ordering: ohe__ first, then s__num_, then num_
    ohe_out = [c for c in feature_df.columns if c.startswith('ohe__')]
    snum_out = [c for c in feature_df.columns if c.startswith('s__num_')]
    num_out = [c for c in feature_df.columns if c.startswith('num_')]
    ordered_cols = ohe_out + snum_out + num_out
    feature_df = feature_df[ordered_cols]

    dataset = pd.concat([outcome_df, feature_df], axis=1)
    print(f"Standard pipeline done. Final shape: {dataset.shape}")
    return dataset