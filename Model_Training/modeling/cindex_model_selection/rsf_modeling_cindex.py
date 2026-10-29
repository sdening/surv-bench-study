import gc
import json
import os
import sys
import time
import yaml
import warnings
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import survhive as sv
from pycox.evaluation import EvalSurv

# survival specific imports

# local project imports
def get_repo_root():
    return Path(__file__).resolve().parent.parent.parent.parent

repo_root = get_repo_root()
EXPERIMENT_RESULTS = repo_root / 'Model_Training' / 'modeling' / 'experiment_results'
sys.path.append(str(repo_root / 'Data_Preprocessing' / 'src'))
sys.path.append(str(repo_root / 'Model_Training' / 'src'))
from modeling_utils import (
    get_time_grid, subsample_time_grid, split_y, load_processed_dataset,
    get_native_surv_from_sksurv, interpolate_survival_to_grid,
    apply_containment_fix,
    compute_baselines, evaluate_on_test, log_cv_summary,
    load_hp_grid,
)
from plot_utils import (
    plot_native_vs_interpolated,
    plot_risk_stratified_km,
    save_calibration_histogram,
    plot_cv_splits,
)

warnings.filterwarnings('ignore', category=UserWarning, module='SurvivalEVAL')

if not hasattr(pd.Series, "is_monotonic"):
    pd.Series.is_monotonic = property(lambda self: self.is_monotonic_increasing)


def optimization_rsf_kfold(
    dataset_name: str,
    seed: int,
    n_folds: int = 5,
    TIME_COL: str = "time",
    EVENT_COL: str = "event",
    fold_time_budget_h: float = 3.0,
    ibs_grid_size: int = None,
    save_plots: bool = False,):
    """
    K-fold CV for RSF:
    - Split data into k folds using StratifiedKFold
    - Per fold: 80% train_full, 20% test
    - Split train_full into 80% train, 20% val
    - Final split per fold: 64% train, 16% val, 20% test
    - Find best HP combo on val (highest C-index)
    - Evaluate that model on test
    - Average performance over k test sets
    """
    
    logger.info(f"K-Fold CV RSF Script started (k={n_folds}) ")
    df = load_processed_dataset(dataset_name, method_name = "rsf", fix_split=False)
    t_min = float(df[TIME_COL].min())
    t_max = float(df[TIME_COL].max())
    _cfg_path = repo_root / 'Data_Preprocessing' / 'configs' / f'{dataset_name}.yaml'
    if not _cfg_path.exists():
        print(f"[WARNING] Dataset config not found: {_cfg_path} — TIME_UNIT defaulting to 'unknown'")
    time_unit = yaml.safe_load(_cfg_path.read_text()).get('TIME_UNIT', 'unknown') if _cfg_path.exists() else 'unknown'

    X_all_df = df.drop(columns=[EVENT_COL, TIME_COL])
    X_all_np = X_all_df.to_numpy().astype("float32", copy=False)
    y_all = np.array(
        list(zip(df[EVENT_COL].astype(bool), df[TIME_COL].astype(float))),
        dtype=[("event", "?"), ("time", "<f8")],
    )
    y_event_all = y_all["event"].astype(int)

    logger.info(f"Dataset {dataset_name}: n={len(df)}, events={int(y_event_all.sum())}, censor_rate={1.0 - float(y_event_all.mean()):.3f}")

    # Load HP grid
    grid_dict, combos = load_hp_grid(repo_root, 'rsf', logger)

    results_dir = EXPERIMENT_RESULTS / 'results' / 'rsf' / 'cindex'
    plots_base = EXPERIMENT_RESULTS / 'plots' / 'rsf' / 'cindex'

    start_time = time.perf_counter()
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_results = []
    fold_indices = {}
    failed_combinations = []  # Track failed HP combinations
    
    FOLD_TIME_BUDGET_S = fold_time_budget_h * 3600
    budget_exceeded = False
    for fold, (trainfull_idx, test_idx) in enumerate(kfold.split(X_all_np, y_event_all), start=1):
        logger.info(f"\n{'='*60}\nFold {fold}/{n_folds} | RSF | C-index | {dataset_name}\n{'='*60}")
        
        trainfull_idx = np.asarray(trainfull_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        
        # Split trainfull into train/val (80/20)
        train_rel_idx, val_rel_idx = train_test_split(
            np.arange(len(trainfull_idx)), 
            test_size=0.2, 
            random_state=seed+fold,
            stratify=y_event_all[trainfull_idx]
        )
        train_idx = trainfull_idx[train_rel_idx]
        val_idx = trainfull_idx[val_rel_idx]
        
        fold_indices[f"fold_{fold}"] = {
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "test_indices": test_idx.tolist()
        }
        
        X_train, y_train = X_all_df.iloc[train_idx], y_all[train_idx]
        X_val, y_val = X_all_df.iloc[val_idx], y_all[val_idx]
        X_test, y_test = X_all_df.iloc[test_idx], y_all[test_idx]
        
        # Containment fix: drop samples with time >= max train time
        val_mask, n_dropped_val, t_max_train = apply_containment_fix(y_train, y_val)
        if n_dropped_val > 0:
            logger.info(f"[Fold {fold}] Containment: dropped {n_dropped_val}/{len(y_val)} val samples (time >= {t_max_train})")
            y_val = y_val[val_mask]
            X_val = X_val.iloc[val_mask]

        test_mask, n_dropped_test, _ = apply_containment_fix(y_train, y_test)
        if n_dropped_test > 0:
            logger.info(f"[Fold {fold}] Containment: dropped {n_dropped_test}/{len(y_test)} test samples (time >= {t_max_train})")
            y_test = y_test[test_mask]
            X_test = X_test.iloc[test_mask]
            test_idx = test_idx[test_mask]
        
        Y_train, D_train = split_y(y_train)
        Y_val, D_val = split_y(y_val)
        Y_test, D_test = split_y(y_test)
        
        n_events_train = int(D_train.sum())
        n_events_val = int(D_val.sum())
        n_events_test = int(D_test.sum())
        
        logger.info(f"Train: {len(train_idx)} ({len(train_idx)/len(X_all_np)*100:.1f}%), events={n_events_train}")
        logger.info(f"Val: {len(val_idx)} ({len(val_idx)/len(X_all_np)*100:.1f}%), events={n_events_val}")
        logger.info(f"Test: {len(test_idx)} ({len(test_idx)/len(X_all_np)*100:.1f}%), events={n_events_test}")
        
        time_grid = get_time_grid(df, y_train, y_val)
        time_grid = subsample_time_grid(time_grid, ibs_grid_size, logger)
        if time_grid is None or len(time_grid) == 0:
            logger.warning(f"Fold {fold}: empty time grid, skipping")
            continue
        
        # HP search
        best_hp_idx = None
        best_val_cindex = float("-inf")
        best_model = None
        
        t_hp_start = time.perf_counter()
        for ci, params in enumerate(combos):
            logger.info(f" Fold {fold} HP {ci+1}/{len(combos)}: {params}")
            
            try:
                # Build and train RSF model
                rsf = sv.RSF(rng_seed=seed + fold, **params)
                rsf.fit(X_train, y_train)
                
                # Evaluate on val: get native predictions, interpolate once to eval grid
                native_times_val, native_surv_val = get_native_surv_from_sksurv(rsf, X_val)
                estimate_val = interpolate_survival_to_grid(native_times_val, native_surv_val, time_grid)
                estimate_val = np.asarray(estimate_val, dtype=float)
                surv_val_df = pd.DataFrame(estimate_val.T, index=time_grid)
                ev_val = EvalSurv(surv_val_df, Y_val.astype(float), D_val.astype(float), steps='post')
                cindex = float(ev_val.concordance_td(method='adj_antolini'))
                
                # Check for NaN, Inf, or zero C-index
                if not np.isfinite(cindex) or cindex == 0:
                    if np.isnan(cindex):
                        reason = "NaN"
                    elif np.isinf(cindex):
                        reason = "Inf"
                    else:
                        reason = "zero"
                    logger.warning(f"Fold {fold} HP {ci+1}: val_C-index is {reason}, skipping")
                    failed_combinations.append({
                        "fold": fold,
                        "hp_index": ci,
                        "params": params,
                        "reason": f"val_C-index is {reason}",
                        "val_cindex": str(cindex)
                    })
                    continue
                
                logger.info(f"Fold {fold} HP {ci+1}/{len(combos)}: val_C-index={cindex:.6f}")
                
                if cindex > best_val_cindex:
                    best_val_cindex = cindex
                    best_hp_idx = ci
                    best_model = rsf
                    
            except Exception as e:
                logger.warning(f"Fold {fold} HP {ci+1} failed: {e}")
                failed_combinations.append({
                    "fold": fold,
                    "hp_index": ci,
                    "params": params,
                    "reason": str(e),
                    "error_type": type(e).__name__
                })
        

            # Time budget check after each HP combo
            elapsed_hp = time.perf_counter() - t_hp_start
            if elapsed_hp > FOLD_TIME_BUDGET_S:
                logger.warning(
                    f"Fold {fold}: TIME BUDGET EXCEEDED "
                    f"({FOLD_TIME_BUDGET_S/3600:.0f}h limit, {elapsed_hp/3600:.1f}h elapsed). "
                    f"Completed {ci+1}/{len(combos)} HP combos. Aborting experiment."
                )
                budget_exceeded = True
                break

        t_hp_end = time.perf_counter()
        fold_hp_search_seconds = t_hp_end - t_hp_start

        if budget_exceeded:
            logger.warning(f"Aborting remaining folds due to time budget.")
            break
        
        if best_hp_idx is None:
            logger.warning(f"Fold {fold}: no valid HP found")
            continue
        
        best_params = combos[best_hp_idx]
        logger.info(f"Fold {fold} best HP: {best_params}, val_C-index={best_val_cindex:.6f}")
        
        # Refit model with best params on full training data (train+val)
        t_refit_start = time.perf_counter()
        X_trainfull = pd.concat([X_train, X_val])
        y_trainfull = np.concatenate([y_train, y_val])
        Y_trainfull, D_trainfull = split_y(y_trainfull)
        
        best_model = sv.RSF(rng_seed=seed + fold, **best_params)
        best_model.fit(X_trainfull, y_trainfull)
        #logger.info(f"Fold {fold}: Refitted model with best HP on train+val data")
        t_refit_end = time.perf_counter()
        fold_refit_seconds = t_refit_end - t_refit_start
        
        # Evaluate on test with best model
        time_grid_test = get_time_grid(df, y_trainfull, y_test)
        time_grid_test = subsample_time_grid(time_grid_test, ibs_grid_size, logger)
        
        # Baselines for reference
        n_test = len(Y_test)
        t_test = time_grid_test.astype(float)
        
        # Random (const. prob of 0.5 everywhere)
        random_ibs, km_ibs, km_mae_po = compute_baselines(
            Y_test, D_test, Y_trainfull, D_trainfull, t_test, n_test, logger, fold
        )
        
        # Test evaluation: get native predictions, interpolate ONCE to eval grid
        t_infer_start = time.perf_counter()
        native_times_test, native_surv_test = get_native_surv_from_sksurv(best_model, X_test)
        surv_on_eval_grid = interpolate_survival_to_grid(
            native_times_test, native_surv_test, time_grid_test,
        )
        surv_on_eval_grid = np.clip(surv_on_eval_grid, 1e-6, 1.0 - 1e-6)
        t_infer_end = time.perf_counter()
        fold_inference_seconds = t_infer_end - t_infer_start
        logger.info(f"Fold {fold} TIMING: HP search={fold_hp_search_seconds:.2f}s ({fold_hp_search_seconds/len(combos):.3f}s/combo), refit/model_load={fold_refit_seconds:.2f}s, inference={fold_inference_seconds:.4f}s")

        test_ibs, test_c, x_cal_test, p_value, bin_stats, mae_po_score = evaluate_on_test(
            surv_on_eval_grid, time_grid_test, Y_test, D_test, Y_trainfull, D_trainfull
        )
        
        if save_plots:
            calib_path = save_calibration_histogram(
                bin_stats, dataset_name, "RSF", fold,
                save_dir=plots_base / 'calibration_histogram' / 'rsf',
                metric_label="cindex",
            )
            logger.info(f"Saved calibration plot: {calib_path}")
            
            risk_scores = best_model.predict(X_test)
            km_path = plot_risk_stratified_km(
                risk_scores, Y_test, D_test, Y_train, D_train,
                dataset_name, "RSF", fold,
                save_dir=plots_base / 'km_curves' / 'rsf',
                metric_label="cindex",
            )
            logger.info(f"Saved KM curve plot: {km_path}")
            
            surv_path = plot_native_vs_interpolated(
                native_times_test, native_surv_test, time_grid_test, surv_on_eval_grid,
                test_idx, dataset_name, "RSF", fold, seed,
                save_dir=plots_base / 'individual_surv' / 'rsf',
                metric_label="cindex",
                native_style="step",
            )
            logger.info(f"Saved native vs interpolated survival curves: {surv_path}")
            plt.close('all')

        logger.info(f"Fold {fold} TEST: IBS={test_ibs:.6f}, C-index={test_c:.6f}, x_cal={x_cal_test:.6f}, p={p_value:.6g}, MAE_PO={mae_po_score:.6f}")
        
        fold_results.append({
            "fold": fold,
            "best_params": best_params,
            "fold_hp_search_seconds": fold_hp_search_seconds,
            "fold_hp_time_per_combo": fold_hp_search_seconds / len(combos),
            "fold_refit_seconds": fold_refit_seconds,
            "fold_inference_seconds": fold_inference_seconds,
            "val_cindex": best_val_cindex,
            "test_ibs": test_ibs,
            "test_c": test_c,
            "test_x_cal": x_cal_test,
            "test_p": float(p_value),
            "random_ibs": random_ibs,
            "km_train_ibs": km_ibs,
            "mae_po": float(mae_po_score),
            "km_train_mae_po": float(km_mae_po),
            "n_dropped_val": n_dropped_val,
            "n_dropped_test": n_dropped_test,
        })
    
    elapsed = time.perf_counter() - start_time
    
    """
    # Create split visualization
    plot_path = plot_cv_splits(
        fold_indices, len(X_all_np), n_folds,
        dataset_name, "RSF",
        save_dir=plots_base / 'cv_splits',
        metric_label="cindex",
    )
    logger.info(f"Saved split visualization: {plot_path}")
    """
    
    # Summary
    (mean_test_ibs, std_test_ibs, mean_test_c, std_test_c,
     mean_random_ibs, mean_km_ibs, mean_mae_po, std_mae_po, mean_km_mae_po) = log_cv_summary(
        fold_results, logger, dataset_name, t_min, t_max, time_unit, n_folds, elapsed, n=len(df)
    )
    
    # Save results
    os.makedirs(results_dir, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = results_dir / f"kfold_rsf_{dataset_name}_{ts}_cindex.json"
    
    details = {
        "dataset": dataset_name,
        "seed": seed,
        "n_folds": n_folds,
        "cindex_method": "pycox_adj_antolini",
        "interpolation": "linear",
        "optimization_metric": "C-index",
        "mean_test_ibs": mean_test_ibs,
        "std_test_ibs": std_test_ibs,
        "mean_test_c": mean_test_c,
        "std_test_c": std_test_c,
        "mean_random_ibs": mean_random_ibs,
        "mean_km_ibs": mean_km_ibs,
        "mean_mae_po": mean_mae_po,
        "mean_km_mae_po": mean_km_mae_po,
        "search_space": grid_dict,
        "hp_combinations": [{"hp_index": i, "params": combo} for i, combo in enumerate(combos)],
        "num_combinations": len(combos),
        "elapsed_seconds": elapsed,
        "fold_results": fold_results,
        "fold_indices": fold_indices,
        "failed_combinations": failed_combinations,
        "num_failed": len(failed_combinations),
        "timestamp": ts,
        "budget_exceeded": budget_exceeded,
        "fold_time_budget_h": fold_time_budget_h,
        "ibs_grid_size": ibs_grid_size,
        "t_min": t_min,
        "t_max": t_max,
        "time_unit": time_unit,
    }
    
    with open(details_path, "w") as f:
        json.dump(details, f, indent=2, default=str, separators=(',', ': '))
    
    logger.info(f"Saved results to: {details_path}")
    
    return details_path, details


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=["ovarian", "breast", "micro.censure", "e1684", "acath", "prostateSurvival", "NSBCD", "nwtco", "nki70", "Z243", "burn", "veteran", "divorce", "UnempDur", "Unemployment", "support2"],)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--fold_time_budget_h', type=float, default=3.0,
                        help='Per-fold time budget in hours (default: 3.0)')
    parser.add_argument('--ibs_grid_size', type=int, default=None,
                        help='Subsample the test time grid to this many points for IBS/MAE evaluation (default: None = full grid)')
    parser.add_argument('--save_plots', action='store_true', default=False)
    args = parser.parse_args()

    seed = 42
    
    for dataset_name in args.datasets:
        logger.info(f"\nProcessing {dataset_name}")
        
        logs_dir = EXPERIMENT_RESULTS / 'logs' / 'rsf' / 'cindex'
        os.makedirs(logs_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(logs_dir / f"rsf_kfold_cindex_{dataset_name}.log", mode="w"),
                logging.StreamHandler()
            ],
            force=True
        )
        
        try:
            optimization_rsf_kfold(
                dataset_name=dataset_name,
                seed=seed,
                n_folds=args.n_folds,
                fold_time_budget_h=args.fold_time_budget_h,
                ibs_grid_size=args.ibs_grid_size,
                save_plots=args.save_plots,
            )
        except Exception as e:
            logger.error(f"Failed: {e}")
        finally:
            gc.collect()
