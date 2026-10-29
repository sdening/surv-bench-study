import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Data_Preprocessing', 'src'))
from utils_preprocessing import *

TIME_COL = "time"
EVENT_COL = "event"
CAT_PREFIX = "fac_"
NUM_PREFIX = "s__num_"
OHE_PREFIX = "ohe__"


def get_N(df) -> int:
    return len(df)


def get_shape(df, time_col: str = TIME_COL, event_col: str = EVENT_COL):
    cols_excl = [c for c in df.columns if c not in (time_col, event_col)]
    return (len(df), len(cols_excl))


def get_dim(df, time_col: str = TIME_COL, event_col: str = EVENT_COL):
    """Returns N, p, N/p ratio, p/N ratio, and shape class (tall/wide/balanced)."""
    N = get_N(df)
    p = get_shape(df, time_col, event_col)[1]
    if N == 0:
        raise ValueError("DataFrame has zero rows.")
    if p == 0:
        raise ValueError("No feature columns found after excluding time/event.")
    ratio_N_over_p = N / p
    ratio_p_over_N = p / N
    tol = 1.5
    if ratio_N_over_p >= tol:
        shape_class = "tall"
    elif ratio_p_over_N >= tol:
        shape_class = "wide"
    else:
        shape_class = "balanced"
    return {"N": N, "p": p, "ratio_N_over_p": ratio_N_over_p, "ratio_p_over_N": ratio_p_over_N, "shape_class": shape_class}


def get_n_categorical_features(df, time_col: str = TIME_COL, event_col: str = EVENT_COL, cat_prefix: str = CAT_PREFIX, verbose: bool = True):
    cat_features = [c for c in df.columns if isinstance(c, str) and c not in (time_col, event_col) and c.startswith(cat_prefix)]
    if verbose:
        print(f"Categorical features (={len(cat_features)}): {cat_features}")
    return len(cat_features), cat_features


def get_n_numerical_features(df, time_col: str = TIME_COL, event_col: str = EVENT_COL, num_prefix: str = NUM_PREFIX, verbose: bool = True):
    num_features = [c for c in df.columns if isinstance(c, str) and c not in (time_col, event_col) and c.startswith(num_prefix)]
    if verbose:
        print(f"Numerical features (={len(num_features)}): {num_features}")
    return len(num_features), num_features


def get_n_ohe_features(df, time_col: str = TIME_COL, event_col: str = EVENT_COL, ohe_prefix: str = OHE_PREFIX, verbose: bool = True):
    ohe_features = [c for c in df.columns if isinstance(c, str) and c not in (time_col, event_col) and c.startswith(ohe_prefix)]
    groups: Dict[str, int] = {}
    for col in ohe_features:
        stripped = col[len(ohe_prefix):]
        parts = stripped.rsplit("_", 1)
        source = parts[0] if len(parts) > 1 else stripped
        groups[source] = groups.get(source, 0) + 1
    if verbose:
        print(f"OHE features (={len(ohe_features)}): {ohe_features}")
        if groups:
            print(f"\tGrouped by source feature: {groups}")
    return len(ohe_features), ohe_features, groups


def get_category_counts_and_distributions(df, event_col: str = EVENT_COL, time_col: str = TIME_COL, cat_prefix: str = CAT_PREFIX, normalize: bool = False, dropna: bool = True, verbose: bool = True):
    _, cats = get_n_categorical_features(df, event_col=event_col, time_col=time_col, cat_prefix=cat_prefix, verbose=verbose)
    n_cats, dists = {}, {}
    for c in cats:
        s = df[c].dropna() if dropna else df[c]
        n_cats[c] = s.nunique(dropna=not dropna)
        dists[c] = s.value_counts(normalize=normalize, dropna=not dropna)
    if verbose:
        mean_cats = float(pd.Series(n_cats).mean()) if n_cats else float("nan")
        print(f"Mean number of categories across ALL categorical features: {mean_cats:.1f}")
        print("Number of categories per feature:", n_cats)
        print()
    return n_cats, dists


def get_censoring_rate_pct(df, event_col: str = EVENT_COL):
    ser = df[event_col].dropna()
    if ser.empty:
        return float("nan")
    return round(100.0 * (1.0 - ser.astype(bool).mean()), 2)


def _plot_histogram(t, title, label, bins, density, show):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(t, bins=bins, density=density, label=label)
    med, mean = float(t.median()), float(t.mean())
    ax.axvline(med, color="black", linestyle="--", label=f"Median = {med:.2f}")
    ax.axvline(mean, color="black", linestyle=":", label=f"Mean = {mean:.2f}")
    ax.set_xlabel("time")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title(title)
    ax.legend()
    if show:
        plt.show()


def plot_time_histogram(df, ds_name, time_col=TIME_COL, event_col=EVENT_COL, bins=30, density=False, show=True):
    t = pd.to_numeric(df[time_col], errors="coerce").dropna()
    t = t[t >= 0]
    _plot_histogram(t, f"Survival Times (all) - {ds_name}", f"All (n={len(t)})", bins, density, show)


def plot_uncensored_time_histogram(df, ds_name, time_col=TIME_COL, event_col=EVENT_COL, bins=30, density=False, show=True):
    t = pd.to_numeric(df.loc[df[event_col] == True, time_col], errors="coerce").dropna()
    t = t[t >= 0]
    _plot_histogram(t, f"Survival Times (uncensored) - {ds_name}", f"Uncensored (n={len(t)} / {len(df)})", bins, density, show)


def complete_analysis(df, ds_name: str, TIME_COL: str = "time", EVENT_COL: str = "event") -> dict:
    print("Examine dataset:")
    print("\t - Columns: ", df.columns.tolist())
    print("\t - # rows:", get_N(df))
    print("\t - shape (excl. time/event):", get_shape(df, event_col=EVENT_COL, time_col=TIME_COL))
    dim = get_dim(df, event_col=EVENT_COL, time_col=TIME_COL)
    print("\t - Dimensions:", dim)
    n_cat, _ = get_n_categorical_features(df, event_col=EVENT_COL, time_col=TIME_COL)
    print("\t - # categorical features:", n_cat)
    n_num, _ = get_n_numerical_features(df, event_col=EVENT_COL, time_col=TIME_COL)
    print("\t - # numerical features:", n_num)
    n_ohe, _, _ = get_n_ohe_features(df, event_col=EVENT_COL, time_col=TIME_COL)
    print("\t - # OHE features:", n_ohe)
    print("Get category counts and distribution:")
    get_category_counts_and_distributions(df, event_col=EVENT_COL, time_col=TIME_COL)
    cens_rate = get_censoring_rate_pct(df, event_col=EVENT_COL)
    n_events = int(df[EVENT_COL].sum())
    print("\t - Censoring rate %:", cens_rate)
    print("\t - # events:", n_events)
    plot_time_histogram(df, event_col=EVENT_COL, time_col=TIME_COL, ds_name=ds_name)
    plot_uncensored_time_histogram(df, event_col=EVENT_COL, time_col=TIME_COL, ds_name=ds_name)

    return {
        "dataset":            ds_name,
        "N":                  dim["N"],
        "p":                  dim["p"],
        "shape_class":        dim["shape_class"],
        "n_cat":              n_cat,
        "n_num":              n_num,
        "n_ohe":              n_ohe,
        "n_events":           n_events,
        "censoring_rate_pct": cens_rate,
        "event_rate_pct":     round(100.0 - cens_rate, 2),
    }
