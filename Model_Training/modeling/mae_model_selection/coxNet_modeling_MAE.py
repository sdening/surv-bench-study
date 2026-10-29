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
from sksurv.linear_model import CoxnetSurvivalAnalysis

# survival specific imports
from SurvivalEVAL.Evaluator import SurvivalEvaluator

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
    apply_containment_fix, compute_baselines, evaluate_on_test, log_cv_summary,
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


def optimization_coxnet_kfold(
    dataset_name: str,
    seed: int,
    n_folds: int = 5,
    TIME_COL: str = "time",
    EVENT_COL: str = "event",
    fold_time_budget_h: float = 3.0,
    ibs_grid_size: int = None,
    save_plots: bool = False,
):
    """
    K-fold CV for CoxNet:
    - Split data into k folds using StratifiedKFold
    - Per fold: 80% train_full, 20% test
    - Split train_full into 80% train, 20% val
    - Final split per fold: 64% train, 16% val, 20% test
    - Find best HP combo on val (lowest IBS)
    - Evaluate that model on test
    - Average performance over k test sets
    """
    
    logger.info(f"K-Fold CV CoxNet Script started (k={n_folds})")
    df = load_processed_dataset(dataset_name, method_name='cox_elasticnet', fix_split=False)
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
    method_config_path = repo_root / 'Model_Training' / 'method_configs' / 'cox_elasticnet.yaml'
    if not method_config_path.exists():
        print(f"[WARNING] Method config not found: {method_config_path}")
    with open(method_config_path, "r") as f:
        method_config = yaml.safe_load(f)

    grid_dict = method_config.get("hyperparameter_grid", {})
    l1_ratios = grid_dict.get("l1_ratio", [0.5])
    logger.info(f"l1_ratio values for HP search: {l1_ratios}")

    results_dir = EXPERIMENT_RESULTS / 'results' / 'coxnet' / 'mae'
    plots_base = EXPERIMENT_RESULTS / 'plots' / 'coxnet' / 'mae'

    start_time = time.perf_counter()
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_results = []
    fold_indices = {}
    failed_combinations = []  # Track failed HP combinations
    failed_folds = []
    
    FOLD_TIME_BUDGET_S = fold_time_budget_h * 3600
    budget_exceeded = False
    for fold, (trainfull_idx, test_idx) in enumerate(kfold.split(X_all_np, y_event_all), start=1):
        logger.info(f"\n{'='*60}\nFold {fold}/{n_folds} | CoxNet | MAE-PO | {dataset_name}\n{'='*60}")
        
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
        
        # Estimate alphas for this fold using scikit-survival directly
        logger.info(f"Estimating alphas for fold {fold}...")
        alphas_per_l1 = {}  # Store alphas per l1_ratio
        coef_dfs_per_l1 = {}
        for l1_est in l1_ratios:
            coxnet_estimator = CoxnetSurvivalAnalysis(l1_ratio=l1_est, alpha_min_ratio=0.01, max_iter=100)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coxnet_estimator.fit(X_train, y_train)
            alphas_full = coxnet_estimator.alphas_.tolist()
            alphas_subsampled = alphas_full[::20]
            alphas_per_l1[l1_est] = alphas_subsampled
            logger.info(f"Fold {fold}, l1_ratio={l1_est}: {len(alphas_full)} alphas -> {len(alphas_subsampled)} subsampled")
            
            # Store coefficients for plotting
            coef_dfs_per_l1[l1_est] = pd.DataFrame(
                coxnet_estimator.coef_, 
                index=X_train.columns, 
                columns=np.round(coxnet_estimator.alphas_, 8)
            )
        
        del coxnet_estimator
        gc.collect()

        if save_plots:
            coef_dir = plots_base / 'coefficient_paths' / 'coxnet' / dataset_name
            os.makedirs(coef_dir, exist_ok=True)
            for l1_est, coef_df in coef_dfs_per_l1.items():
                fig, ax = plt.subplots(figsize=(9, 6))
                alphas_plot = coef_df.columns
                for row in coef_df.itertuples():
                    ax.semilogx(alphas_plot, row[1:], ".-", linewidth=0.8, markersize=3)
                
                alpha_min = alphas_plot.min()
                n_highlight = min(5, len(coef_df))
                top_coefs = coef_df.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
                for name in top_coefs.index:
                    coef = coef_df.loc[name, alpha_min]
                    ax.text(alpha_min, coef, name + "   ", horizontalalignment="right", verticalalignment="center", fontsize=8)
                
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("alpha")
                ax.set_ylabel("coefficient")
                ax.set_title(f"Coefficient Paths (l1_ratio={l1_est}) - {dataset_name} - Fold {fold}")
                
                plt.tight_layout()
                ts_plot = datetime.now().strftime("%Y%m%d_%H%M%S")
                coef_path = coef_dir / f"coef_path_{dataset_name}_l1_{l1_est}_fold{fold}_{ts_plot}.png"
                plt.savefig(coef_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
        
        # Create HP grid: each l1_ratio paired with its own estimated alphas
        combos = []
        for l1 in l1_ratios:
            for alpha in alphas_per_l1[l1]:
                combos.append({"l1_ratio": l1, "alpha": alpha})
        
        logger.info(f"Fold {fold} HP grid size: {len(combos)} combinations")
        
        # HP search
        t_hp_start = time.perf_counter()
        best_hp_idx = None
        best_val_mae_po = float("inf")
        alpha_ibs_results = []
        valid_hp_results = []
        
        for ci, params in enumerate(combos):
            logger.info(f" Fold {fold} HP {ci+1}/{len(combos)}: {params}")
            
            coxnet = None
            try:
                coxnet = sv.CoxNet(rng_seed=seed + fold, **params)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    coxnet.fit(X_train, y_train)
                
                native_times_v, native_surv_v = get_native_surv_from_sksurv(coxnet, X_val)
                estimate_val = interpolate_survival_to_grid(native_times_v, native_surv_v, time_grid)
                
                if not np.all(np.isfinite(estimate_val)):
                    logger.warning(f"Fold {fold} HP {ci+1}: predictions contain NaN/Inf, skipping")
                    failed_combinations.append({
                        "fold": fold,
                        "hp_index": ci,
                        "params": params,
                        "reason": "predictions contain NaN/Inf",
                        "num_nan": int(np.isnan(estimate_val).sum()),
                        "num_inf": int(np.isinf(estimate_val).sum())
                    })
                    continue
                
                ev = SurvivalEvaluator(
                    pred_survs=estimate_val,
                    time_coordinates=time_grid.astype(float),
                    event_times=Y_val,
                    event_indicators=D_val,
                    train_event_times=Y_train,
                    train_event_indicators=D_train,
                )
                
                mae_po = ev.mae(method="Pseudo_obs", weighted=True)
                
                if not np.isfinite(mae_po):
                    reason = "NaN" if np.isnan(mae_po) else "Inf"
                    logger.warning(f"Fold {fold} HP {ci+1}: val_MAE_PO is {reason}, skipping")
                    failed_combinations.append({
                        "fold": fold,
                        "hp_index": ci,
                        "params": params,
                        "reason": f"val_MAE_PO is {reason}",
                        "val_mae_po": str(mae_po)
                    })
                    continue
                
                logger.info(f"Fold {fold} HP {ci+1}/{len(combos)}: val_MAE_PO={mae_po:.6f}")
                
                alpha_ibs_results.append({"alpha": params["alpha"], "l1_ratio": params["l1_ratio"], "mae_po": mae_po})
                valid_hp_results.append({"params": params, "mae_po": mae_po})
                
                del native_times_v, native_surv_v, estimate_val, ev

                if mae_po < best_val_mae_po:
                    best_val_mae_po = mae_po
                    best_hp_idx = ci
                    
            except Exception as e:
                logger.warning(f"Fold {fold} HP {ci+1} failed: {e}")
                failed_combinations.append({
                    "fold": fold,
                    "hp_index": ci,
                    "params": params,
                    "reason": str(e),
                    "error_type": type(e).__name__
                })
            finally:
                del coxnet
                gc.collect()

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
        
        # Sort valid HP results by MAE-PO (ascending = best first)
        valid_hp_results.sort(key=lambda x: x["mae_po"])
        
        best_params = combos[best_hp_idx]
        logger.info(f"Fold {fold} best HP: {best_params}, val_MAE_PO={best_val_mae_po:.6f}")
        
        if save_plots:
            alpha_perf_dir = plots_base / 'alpha_performance' / 'coxnet' / dataset_name
            os.makedirs(alpha_perf_dir, exist_ok=True)
            
            best_l1 = best_params["l1_ratio"]
            l1_results = [r for r in alpha_ibs_results if r["l1_ratio"] == best_l1]
            alphas_l1 = [r["alpha"] for r in l1_results]
            mae_po_l1 = [r["mae_po"] for r in l1_results]
            
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(alphas_l1, mae_po_l1, "o-", linewidth=2, markersize=5)
            ax.set_xscale("log")
            ax.set_ylabel("Val MAE-PO")
            ax.set_xlabel("alpha")
            ax.axvline(best_params["alpha"], c="orange", linestyle="--", linewidth=2, label=f"Selected: α={best_params['alpha']:.4f}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_title(f"Alpha vs MAE-PO (l1_ratio={best_l1}) - {dataset_name} - Fold {fold}")
            plt.tight_layout()
            ts_plot = datetime.now().strftime("%Y%m%d_%H%M%S")
            alpha_perf_path = alpha_perf_dir / f"alpha_perf_{dataset_name}_fold{fold}_mae_{ts_plot}.png"
            plt.savefig(alpha_perf_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        # Refit model with best HP on train+val, with fallback to next-best if it fails
        X_trainfull = pd.concat([X_train, X_val])
        y_trainfull = np.concatenate([y_train, y_val])
        Y_trainfull, D_trainfull = split_y(y_trainfull)
        
        t_refit_start = time.perf_counter()
        retrain_success = False
        best_model = None
        
        for i, hp_result in enumerate(valid_hp_results):
            params = hp_result["params"]
            val_mae_po = hp_result["mae_po"]
            
            logger.info(f"Fold {fold} trying HP rank {i+1}: {params}, val_MAE_PO={val_mae_po:.6f}")
            
            try:
                best_model = sv.CoxNet(rng_seed=seed + fold, **params)
                best_model.fit(X_trainfull, y_trainfull)
                retrain_success = True
                best_params = params
                logger.info(f"Fold {fold}: Retraining succeeded with rank {i+1} HP")
                break
                
            except Exception as e:
                if "weights are too large" in str(e) or "Numerical error" in str(e):
                    logger.warning(f"Fold {fold}: Retraining failed with rank {i+1} HP: {e}")
                    continue
                else:
                    raise e
        
        t_refit_end = time.perf_counter()
        fold_refit_seconds = t_refit_end - t_refit_start
        
        if not retrain_success:
            logger.error(f"Fold {fold}: All retraining attempts failed, skipping fold")
            continue
        
        # Feature importance from the retrained model's coefficients
        sksurv_model = getattr(best_model, "model_", best_model)
        best_coefs = pd.DataFrame(sksurv_model.coef_, index=X_trainfull.columns, columns=["coefficient"])
        non_zero_coefs = best_coefs.query("coefficient != 0")
        non_zero = len(non_zero_coefs)
        logger.info(f"Fold {fold}: {non_zero} non-zero coefficients")
        
        numeric_cols = [c for c in non_zero_coefs.index if X_trainfull[c].nunique() > 2]
        non_zero_numeric = non_zero_coefs.loc[numeric_cols]
        n_excluded = non_zero - len(non_zero_numeric)
        if n_excluded > 0:
            logger.info(f"Fold {fold}: excluded {n_excluded} binary/categorical features from importance plot")
        n_plot = len(non_zero_numeric)
        
        if n_plot > 0 and save_plots:
            coef_order = non_zero_numeric.abs().sort_values("coefficient").index
            fig, ax = plt.subplots(figsize=(6, min(8, max(4, n_plot * 0.3))))
            non_zero_numeric.loc[coef_order].plot.barh(ax=ax, legend=False)
            ax.set_xlabel("coefficient")
            ax.set_title(f"Feature Importance (numerical only) - {dataset_name} - Fold {fold}")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            feat_imp_dir = plots_base / 'feature_importance' / 'coxnet' / dataset_name
            os.makedirs(feat_imp_dir, exist_ok=True)
            ts_plot = datetime.now().strftime("%Y%m%d_%H%M%S")
            feat_imp_path = feat_imp_dir / f"feat_imp_{dataset_name}_fold{fold}_{ts_plot}.png"
            plt.savefig(feat_imp_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved feature importance plot: {feat_imp_path}")
        
        # Evaluate on test with best model
        time_grid_test = get_time_grid(df, y_trainfull, y_test)
        time_grid_test = subsample_time_grid(time_grid_test, ibs_grid_size, logger)

        # baseleines for references
        n_test = len(Y_test)
        t_test = time_grid_test.astype(float)

        random_ibs, km_ibs, km_mae_po = compute_baselines(
            Y_test, D_test, Y_trainfull, D_trainfull, t_test, n_test, logger, fold
        )

        # Test evaluation: get native predictions, interpolate to eval grid
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
                bin_stats, dataset_name, "CoxNet", fold,
                save_dir=plots_base / 'calibration_histogram' / 'coxnet',
                metric_label="mae",
            )
            logger.info(f"Saved calibration plot: {calib_path}")
            
            risk_scores = best_model.predict(X_test)
            km_path = plot_risk_stratified_km(
                risk_scores, Y_test, D_test, Y_train, D_train,
                dataset_name, "CoxNet", fold,
                save_dir=plots_base / 'km_curves' / 'coxnet',
                metric_label="mae",
            )
            logger.info(f"Saved KM curve plot: {km_path}")
            
            surv_path = plot_native_vs_interpolated(
                native_times_test, native_surv_test, time_grid_test, surv_on_eval_grid,
                test_idx, dataset_name, "CoxNet", fold, seed,
                save_dir=plots_base / 'individual_surv' / 'coxnet',
                metric_label="mae",
                native_style="step",
            )
            logger.info(f"Saved native vs interpolated survival curves: {surv_path}")
            logger.info(f"Saved coefficient path plots: {coef_dir}")
            logger.info(f"Saved alpha performance plots: {alpha_perf_dir}")
        
        logger.info(f"Fold {fold} TEST: IBS={test_ibs:.6f}, C-index={test_c:.6f}, x_cal={x_cal_test:.6f}, MAE-PO={mae_po_score:.6f}, p={p_value:.6g}")
        
        fold_results.append({
            "fold": fold,
            "best_params": best_params,
            "alphas_per_l1": {str(k): v for k, v in alphas_per_l1.items()},
            "n_dropped_val": n_dropped_val,
            "n_dropped_test": n_dropped_test,
            "fold_hp_search_seconds": fold_hp_search_seconds,
            "fold_hp_time_per_combo": fold_hp_search_seconds / len(combos),
            "fold_refit_seconds": fold_refit_seconds,
            "fold_inference_seconds": fold_inference_seconds,
            "val_mae_po": best_val_mae_po,
            "test_ibs": test_ibs,
            "test_c": test_c,
            "test_x_cal": x_cal_test,
            "test_p": float(p_value),
            "random_ibs": random_ibs,
            "km_train_ibs": km_ibs,
            "mae_po": float(mae_po_score),
            "km_train_mae_po": float(km_mae_po),
        })

        del best_model, sksurv_model, X_trainfull, y_trainfull
        del surv_on_eval_grid, native_times_test, native_surv_test
        del coef_dfs_per_l1, valid_hp_results, alpha_ibs_results
        gc.collect()
    
    elapsed = time.perf_counter() - start_time
    
    """
    plot_path = plot_cv_splits(
        fold_indices, len(X_all_np), n_folds,
        dataset_name, "CoxNet",
        save_dir=plots_base / 'cv_splits',
        metric_label="mae",
    )
    logger.info(f"Saved split visualization: {plot_path}")
    """
    
    x_cal_vals = [r["test_x_cal"] for r in fold_results if np.isfinite(r["test_x_cal"])]
    mean_test_x_cal = float(np.mean(x_cal_vals)) if x_cal_vals else float("nan")
    std_test_x_cal = float(np.std(x_cal_vals)) if x_cal_vals else float("nan")

    (mean_test_ibs, std_test_ibs, mean_test_c, std_test_c,
     mean_random_ibs, mean_km_ibs, mean_mae_po, std_mae_po, mean_km_mae_po) = log_cv_summary(
        fold_results, logger, dataset_name, t_min, t_max, time_unit, n_folds, elapsed, n=len(df)
    )

    os.makedirs(results_dir, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = results_dir / f"kfold_coxnet_{dataset_name}_{ts}_mae.json"
    
    details = {
        "dataset": dataset_name,
        "t_min": t_min,
        "t_max": t_max,
        "time_unit": time_unit,
        "seed": seed,
        "n_folds": n_folds,
        "cindex_method": "pycox_adj_antolini",
        "interpolation": "linear",
        "optimization_metric": "MAE-PO",
        "mean_test_ibs": mean_test_ibs,
        "std_test_ibs": std_test_ibs,
        "mean_test_c": mean_test_c,
        "std_test_c": std_test_c,
        "mean_random_ibs": mean_random_ibs,
        "mean_km_ibs": mean_km_ibs,
        "mean_test_mae_po": mean_mae_po,
        "std_test_mae_po": std_mae_po,
        "mean_km_mae_po": mean_km_mae_po,
        "mean_test_x_cal": mean_test_x_cal,
        "std_test_x_cal": std_test_x_cal,
        "search_space": grid_dict,
        "hp_combinations": [{"hp_index": i, "params": combo} for i, combo in enumerate(combos)],
        "num_combinations": len(combos),
        "elapsed_seconds": elapsed,
        "fold_results": fold_results,
        "fold_indices": fold_indices if len(df) <= 10_000 else "omitted (n > 10k)",
        "failed_combinations": failed_combinations,
        "num_failed": len(failed_combinations),
        "timestamp": ts,
        "budget_exceeded": budget_exceeded,
        "fold_time_budget_h": fold_time_budget_h,
        "ibs_grid_size": ibs_grid_size,
        "failed_folds": failed_folds,
    }
    
    with open(details_path, "w") as f:
        json.dump(details, f, indent=2, default=str, separators=(',', ': '))
    
    logger.info(f"Saved results to: {details_path}")
    
    return details_path, details


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', 
                        default=["ovarian", "breast", "micro.censure", "e1684", "acath", "prostateSurvival", "NSBCD", "nwtco", "nki70", "Z243", "burn", "veteran", "divorce", "UnempDur", "Unemployment", "support2"],
                        )
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--save_plots', action='store_true', default=False)
    parser.add_argument('--fold_time_budget_h', type=float, default=3.0,
                        help='Per-fold time budget in hours (default: 3.0)')
    parser.add_argument('--ibs_grid_size', type=int, default=None,
                        help='Subsample IBS time grid to this many evenly-spaced points (default: None = no subsampling)')
    args = parser.parse_args()

    seed = 42
    
    for dataset_name in args.datasets:
        logger.info(f"\nProcessing {dataset_name}")
        
        logs_dir = EXPERIMENT_RESULTS / 'logs' / 'coxnet' / 'mae'
        os.makedirs(logs_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(logs_dir / f"coxnet_kfold_ibs_{dataset_name}.log", mode="w"),
                logging.StreamHandler()
            ],
            force=True
        )
        
        try:
            optimization_coxnet_kfold(
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
