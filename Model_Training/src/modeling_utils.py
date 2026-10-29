import itertools
import pandas as pd
try:
    import survhive as sv
except ImportError:
    sv = None
import numpy as np
import yaml
import torch
from pathlib import Path
import sys
from pycox.evaluation import EvalSurv
from SurvivalEVAL.Evaluator import SurvivalEvaluator
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.functions import StepFunction
from evaluation_utils import breslow_fcn, surv_df_from_aft_mu

def get_repo_root():
    """Get the repository root directory."""
    return Path(__file__).resolve().parent.parent.parent


 



def split_y(y):
    """ Split survival data into time and event arrays."""
    y_arr = y
    if isinstance(y_arr, pd.DataFrame):
        return y_arr["time"].to_numpy(dtype="float32"), y_arr["event"].to_numpy(dtype="int32")
    y_arr = np.asarray(y_arr)
    if y_arr.dtype.names and {"time","event"}.issubset(y_arr.dtype.names):
        return y_arr["time"].astype("float32"), y_arr["event"].astype("int32")
    t = np.asarray([row["time"]  if isinstance(row, dict) else row[1] for row in y_arr], dtype="float32")
    e = np.asarray([row["event"] if isinstance(row, dict) else row[0] for row in y_arr], dtype="int32")
    return t, e



def interpolate_survival_to_grid(
    native_times: np.ndarray,
    native_surv: np.ndarray,
    eval_grid: np.ndarray,
) -> np.ndarray:
    """Standardized linear interpolation of survival curves onto an evaluation grid. -- inspired by "Bridging the Gap: Improve Neural Survival Models with Interpolation Techniques" Archetti et al. (2024)"""
    native_times = np.asarray(native_times, dtype=float)
    eval_grid = np.asarray(eval_grid, dtype=float)
    native_surv = np.asarray(native_surv, dtype=float)

    n_samples = native_surv.shape[0]

    # Left boundary: (0, 1.0) if first native time > 0
    if native_times[0] > 0:
        native_times = np.concatenate([[0.0], native_times])
        native_surv = np.column_stack([
            np.ones(n_samples), native_surv,
        ])

    # Right boundary: append far-future anchor (10 * t_last, 0.0)
    t_far = native_times[-1] * 10.0
    native_times = np.concatenate([native_times, [t_far]])
    native_surv = np.column_stack([native_surv, np.zeros(n_samples)])

    # Clip any out-of-range native values
    native_surv = np.clip(native_surv, 0.0, 1.0)

    surv_interp = np.empty((n_samples, len(eval_grid)), dtype=float)
    for i in range(n_samples):
        surv_interp[i] = np.interp(eval_grid, native_times, native_surv[i])

    return surv_interp


def get_native_surv_from_sksurv(model, X):
    """Extract native survival predictions from a fitted sksurv model"""
    sksurv_model = getattr(model, "model_", model)
    step_fns = sksurv_model.predict_survival_function(X)
    native_times = step_fns[0].x
    native_surv = np.column_stack([fn.y for fn in step_fns]).T
    return native_times, native_surv


def get_native_surv_coxkan(model, X_train_np, Y_train, D_train, X_pred_np,
                           chunk_size=2000):
    """Compute native survival predictions via Breslow estimator."""
    log_ph_parts = []
    for i in range(0, len(X_train_np), chunk_size):
        with torch.no_grad():
            part = model(
                torch.tensor(X_train_np[i:i + chunk_size], dtype=torch.float32)
            ).numpy().flatten()
        log_ph_parts.append(part)
    log_ph_train = np.clip(np.concatenate(log_ph_parts), -500, 500)

    event_times, H0 = breslow_fcn(Y_train, D_train, log_ph_train)

    n_times = len(H0)
    S = np.empty((len(X_pred_np), n_times), dtype=np.float64)
    for i in range(0, len(X_pred_np), chunk_size):
        with torch.no_grad():
            lph = model(
                torch.tensor(X_pred_np[i:i + chunk_size], dtype=torch.float32)
            ).numpy().flatten()
        lph = np.clip(lph, -500, 500)
        S[i:i + chunk_size] = np.exp(-np.outer(np.exp(lph), H0))

    return np.asarray(event_times, dtype=float), S


def get_native_surv_xgb_cox(y_train, lin_pred_train, lin_pred_eval):
    """Breslow estimator: linear predictor -> S(t) = S0(t)^exp(lp)."""
    t_ev, H0 = breslow_fcn(
        np.asarray(y_train["time"], float),
        np.asarray(y_train["event"], bool),
        lin_pred_train,
    )
    S0_ev = np.exp(-H0)
    ph_eval = np.exp(lin_pred_eval)
    native_surv = (S0_ev[:, None] ** ph_eval[None, :]).T
    return t_ev, native_surv


def get_native_surv_xgb_aft(preds_mu_log, y_train, dist="normal", scale=1.0):
    """AFT distributional conversion: log(T) prediction -> S(t)."""
    native_times = np.sort(np.unique(np.asarray(y_train["time"], float)))
    native_surv = surv_df_from_aft_mu(preds_mu_log, native_times, dist=dist, scale=scale)
    return native_times, native_surv


def load_hp_grid(repo_root, config_stem, logger):
    """Load HP grid from method config YAML and return (grid_dict, combos)."""
    method_config_path = repo_root / 'Model_Training' / 'method_configs' / f'{config_stem}.yaml'
    if not method_config_path.exists():
        print(f"[WARNING] Method config not found: {method_config_path}")
    with open(method_config_path, 'r') as f:
        method_config = yaml.safe_load(f)

    grid_dict = method_config.get('hyperparameter_grid', {})

    keys = list(grid_dict.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid_dict[k] for k in keys])]
    logger.info(f"HP grid size: {len(combos)}")
    return grid_dict, combos


def get_time_grid(df, y_train, y_test, q_low=1, q_high=99):
    #lower, upper = np.percentile(df["time"], [q_low, q_high]) #percentile as in sci kit survival docu ("data driven": https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html#Time-dependent-Brier-Score)
    #unique_times = np.sort(np.unique(df["time"])) #unique times

    # Handle both pandas DataFrame and structured numpy array formats
    if hasattr(y_train, "time") and hasattr(y_train, "columns"):
        # pandas DataFrame
        train_times = y_train["time"]
        test_times = y_test["time"]
    elif hasattr(y_train, "dtype") and y_train.dtype.names and "time" in y_train.dtype.names:
        # structured numpy array
        train_times = y_train["time"]
        test_times = y_test["time"]
    else:
        raise ValueError("y_train and y_test must be either pandas DataFrames with 'time' column or structured numpy arrays with 'time' field")

    lower, upper = np.percentile(train_times, [q_low, q_high])
    unique_times = np.sort(np.unique(train_times))
    
    #trim grid to upper, lower
    grid = unique_times[(unique_times >= lower) & (unique_times <= upper)]
    
    t_max_train = train_times.max()
    t_max_test  = test_times.max()
    t_min_test = test_times.min()
    
    t_low  = max(lower, t_min_test)
    #print("Lower percentile limit: ", lower)
    #print(f"Min test follow-up: {t_min_test}")
    #print("\tt_low set for grid: ", t_low)
    t_high = min(
        upper, #limits should be included -> as IBS is unstable at tails where few people remain at risk
        t_max_train, #IPCW weights, otherwise censoring probability undefined
        t_max_test #cannot compare predicted survival at times that the test set never observed
    )
    #print("Upper percentile limit: ", upper)
    #print(f"Max train follow-up: {t_max_train}")
    #print(f"Max test follow-up:  {t_max_test}")
    #print("\tt_high set for grid: ", t_high)
    #print("\n")
    
    grid = grid[(grid >= t_low) & (grid < t_high)] #trim as every time point in times must lie inside the test follow-up window
    
    #print(f"Grid size: {len(grid)} time points")
    #if len(grid) > 0:
    #    print(f"\t-> Grid range: [{grid[0]:.3f}, {grid[-1]:.3f}], Grid density: {len(grid)/(grid[-1]-grid[0]):.1f} points/unit time")
    #    print(f"\t-> t_high (eval horizon): {t_high:.3f} | max_train={t_max_train:.3f} | max_test={t_max_test:.3f}")
    #else:
    #    print("\tWARNING: Empty grid!")

    return grid.astype(float)


def apply_containment_fix(y_train, y_other, time_col="time"):
    t_max_train = float(y_train[time_col].max())
    keep_mask = y_other[time_col] < t_max_train
    n_dropped = int(np.sum(~keep_mask))
    return keep_mask, n_dropped, t_max_train



def load_processed_dataset(dataset_name: str, method_name: str, seed: int = None, fix_split: bool = False, min_events_threshold: int = 100):
    """Load preprocessed dataset
    """
    repo_root = get_repo_root()
    data_path = repo_root / 'Data_Preprocessing' / 'data' / 'processed' / dataset_name / f'{dataset_name}_processed.csv'
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {data_path}\n"
            f"Run preprocessing for '{dataset_name}' before training."
        )
    
    print(f"Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    
    # Load dataset config to get correct column names and rename to standard 'time' and 'event'
    dataset_config_path = repo_root / 'Data_Preprocessing' / 'configs' / f'{dataset_name}.yaml'
    if dataset_config_path.exists():
        with open(dataset_config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        time_col = dataset_config.get('TIME_COL', 'time')
        event_col = dataset_config.get('EVENT_COL', 'event')
        
        # Rename to standard names if different
        if time_col in df.columns and time_col != 'time':
            df = df.rename(columns={time_col: 'time'})
        if event_col in df.columns and event_col != 'event':
            df = df.rename(columns={event_col: 'event'})
    
    return df








def subsample_time_grid(time_grid, max_points, logger=None):
    """Subsample time grid to at most max_points evenly spaced points.

    Returns the original grid unchanged when max_points is None or the grid
    is already small enough..
    """
    if max_points is not None and len(time_grid) > max_points:
        if logger:
            logger.info(
                f"Subsampling IBS time grid from {len(time_grid)} "
                f"to {max_points} evenly-spaced points"
            )
        return np.linspace(time_grid[0], time_grid[-1], max_points)
    return time_grid


def compute_baselines(Y_test, D_test, Y_trainfull, D_trainfull, t_test, n_test, logger, fold):
    """Compute random and KM baseline metrics for a single CV fold."""
    random_surv = 0.5 * np.ones((n_test, len(t_test)), dtype=float)
    km_t, km_s = kaplan_meier_estimator(D_trainfull.astype(bool), Y_trainfull.astype(float))
    km_func = StepFunction(km_t, km_s)
    km_surv = np.tile(km_func(t_test), (n_test, 1))

    ev_random = SurvivalEvaluator(
        pred_survs=random_surv, time_coordinates=t_test,
        event_times=Y_test, event_indicators=D_test,
        train_event_times=Y_trainfull, train_event_indicators=D_trainfull,
    )
    random_ibs = float(ev_random.integrated_brier_score(
        num_points=None, target_times=t_test, IPCW_weighted=True,
        integration_method="trapz", draw_figure=False,
    ))

    ev_km = SurvivalEvaluator(
        pred_survs=km_surv, time_coordinates=t_test,
        event_times=Y_test, event_indicators=D_test,
        train_event_times=Y_trainfull, train_event_indicators=D_trainfull,
    )
    km_ibs = float(ev_km.integrated_brier_score(
        num_points=None, target_times=t_test, IPCW_weighted=True,
        integration_method="trapz", draw_figure=False,
    ))
    km_mae_po = float(ev_km.mae(method="Pseudo_obs", weighted=True))

    logger.info(
        f"Fold {fold} BASELINES: Random_IBS={random_ibs:.6f}, "
        f"KM(train)_IBS={km_ibs:.6f}, KM(train)_MAE_PO={km_mae_po:.6f}"
    )
    return random_ibs, km_ibs, km_mae_po


def evaluate_on_test(surv_on_eval_grid, time_grid_test, Y_test, D_test, Y_trainfull, D_trainfull):
    """Compute all four test metrics: IBS, C-index, X-cal, MAE-PO."""
    ev_test = SurvivalEvaluator(
        pred_survs=surv_on_eval_grid,
        time_coordinates=time_grid_test.astype(float),
        event_times=Y_test,
        event_indicators=D_test,
        train_event_times=Y_trainfull,
        train_event_indicators=D_trainfull,
    )
    test_ibs = float(ev_test.integrated_brier_score(
        num_points=None,
        target_times=time_grid_test.astype(float),
        IPCW_weighted=True,
        integration_method="trapz",
        draw_figure=False,
    ))
    surv_eval_df = pd.DataFrame(surv_on_eval_grid.T, index=time_grid_test)
    ev_pycox = EvalSurv(surv_eval_df, Y_test.astype(float), D_test.astype(float), steps='post')
    test_c = float(ev_pycox.concordance_td(method='adj_antolini'))
    p_value, bin_stats = ev_test.d_calibration(return_details=True)
    x_cal_test = float(bin_stats["x_calibration"])
    mae_po_score = float(ev_test.mae(method="Pseudo_obs", weighted=True))
    return test_ibs, test_c, x_cal_test, p_value, bin_stats, mae_po_score







def log_cv_summary(fold_results, logger, dataset_name, t_min, t_max, time_unit,
                   n_folds, elapsed, n=None, label=""):
    """Log CV summary statistics and return them as a tuple."""
    test_ibs_vals = [r["test_ibs"] for r in fold_results if np.isfinite(r["test_ibs"])]
    test_c_vals = [r["test_c"] for r in fold_results if np.isfinite(r["test_c"])]
    random_ibs_vals = [r["random_ibs"] for r in fold_results if np.isfinite(r["random_ibs"])]
    km_ibs_vals = [r["km_train_ibs"] for r in fold_results if np.isfinite(r["km_train_ibs"])]
    mae_po_vals = [r["mae_po"] for r in fold_results if np.isfinite(r["mae_po"])]
    km_mae_po_vals = [r["km_train_mae_po"] for r in fold_results if np.isfinite(r["km_train_mae_po"])]

    mean_test_ibs = float(np.mean(test_ibs_vals)) if test_ibs_vals else float("nan")
    std_test_ibs = float(np.std(test_ibs_vals)) if len(test_ibs_vals) > 1 else float("nan")
    mean_test_c = float(np.mean(test_c_vals)) if test_c_vals else float("nan")
    std_test_c = float(np.std(test_c_vals)) if len(test_c_vals) > 1 else float("nan")
    mean_random_ibs = float(np.mean(random_ibs_vals)) if random_ibs_vals else float("nan")
    mean_km_ibs = float(np.mean(km_ibs_vals)) if km_ibs_vals else float("nan")
    mean_mae_po = float(np.mean(mae_po_vals)) if mae_po_vals else float("nan")
    std_mae_po = float(np.std(mae_po_vals)) if len(mae_po_vals) > 1 else float("nan")
    mean_km_mae_po = float(np.mean(km_mae_po_vals)) if km_mae_po_vals else float("nan")

    label_str = f" {label}" if label else ""
    n_str = f" | N={n}" if n is not None else ""
    logger.info(f"\n{'='*60}")
    logger.info(f"K-FOLD CV SUMMARY{label_str} ({n_folds} folds)")
    logger.info(f"Dataset: {dataset_name} | Time range: [{t_min:.3f}, {t_max:.3f}] {time_unit}{n_str}")
    logger.info(f"Mean Test IBS: {mean_test_ibs:.6f} +/- {std_test_ibs:.6f}")
    logger.info(f"Mean Test C-index: {mean_test_c:.6f} +/- {std_test_c:.6f}")
    logger.info(f"Mean Test MAE-PO: {mean_mae_po:.6f} +/- {std_mae_po:.6f}")
    logger.info(f"Mean Random IBS: {mean_random_ibs:.6f}")
    logger.info(f"Mean KM(train) IBS: {mean_km_ibs:.6f}")
    logger.info(f"Mean KM(train) MAE-PO: {mean_km_mae_po:.6f}")
    if np.isfinite(mean_random_ibs) and mean_random_ibs > 0:
        logger.info(f"Improvement over Random: {(mean_random_ibs - mean_test_ibs) / mean_random_ibs * 100:.1f}%")
    if np.isfinite(mean_km_ibs) and mean_km_ibs > 0:
        logger.info(f"Improvement over KM: {(mean_km_ibs - mean_test_ibs) / mean_km_ibs * 100:.1f}%")
    if np.isfinite(mean_km_mae_po) and mean_km_mae_po > 0:
        logger.info(f"MAE-PO Improvement over KM: {(mean_km_mae_po - mean_mae_po) / mean_km_mae_po * 100:.1f}%")
    logger.info(f"Total time: {elapsed:.2f}s")
    logger.info(f"{'='*60}")

    return (mean_test_ibs, std_test_ibs, mean_test_c, std_test_c,
            mean_random_ibs, mean_km_ibs, mean_mae_po, std_mae_po, mean_km_mae_po)