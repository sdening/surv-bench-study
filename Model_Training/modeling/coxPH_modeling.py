import gc
import itertools
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
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.functions import StepFunction

# survival specific imports
from SurvivalEVAL.Evaluator import SurvivalEvaluator

# local project imports
def get_repo_root():
    return Path(__file__).resolve().parent.parent.parent

repo_root = get_repo_root()
EXPERIMENT_RESULTS = repo_root / 'Model_Training' / 'modeling' / 'experiment_results'
sys.path.append(str(repo_root / 'Data_Preprocessing' / 'src'))
sys.path.append(str(repo_root / 'Model_Training' / 'src'))
from pycox.evaluation import EvalSurv
from modeling_utils import (
    get_time_grid, subsample_time_grid, split_y, load_processed_dataset,
    get_native_surv_from_sksurv, interpolate_survival_to_grid,
    apply_containment_fix,
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


def optimization_coxph_kfold(
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
    K-fold CV for CoxPH...

    NOTE!:
    CoxPH has no hyperparameters to tune (alpha=0.0, ties='efron' are fixed).
    There is no HP selection step — the model is fit once per fold and evaluated
    on the held-out test set.  All four metrics are reported as fold means:
        - Antolini time-dependent C-index C^td
        - IBS (Integrated Brier score)
        - MAE-PO (pseudo-observation mean absolute error)
        - X-CAL

    Split per fold:  64% train / 16% val / 20% test  (val kept for time-grid
    consistency with other scripts but it does not drive any model selection).
    """
    logger.info(f"K-Fold CV CoxPH started (k={n_folds}) — no HP tuning, all metrics reported as fold means")
    df = load_processed_dataset(dataset_name, method_name = "coxph", fix_split=False)
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

    # Load fixed model config (alpha=0.0, ties='efron' — no real HP search)
    method_config_path = repo_root / 'Model_Training' / 'method_configs' / 'coxph.yaml'
    if not method_config_path.exists():
        print(f"[WARNING] Method config not found: {method_config_path}")
    with open(method_config_path, "r") as f:
        method_config = yaml.safe_load(f)

    grid_dict = method_config.get("hyperparameter_grid", {})
    keys = list(grid_dict.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid_dict[k] for k in keys])]
    logger.info(f"Fixed model params: {combos[0]}")

    results_dir = EXPERIMENT_RESULTS / 'results' / 'coxph'
    plots_base = EXPERIMENT_RESULTS / 'plots' / 'coxph'

    start_time = time.perf_counter()
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_results = []
    fold_indices = {}
    failed_combinations = []  # Track failed HP combinations
    failed_folds = []
    
    FOLD_TIME_BUDGET_S = fold_time_budget_h * 3600
    budget_exceeded = False
    for fold, (trainfull_idx, test_idx) in enumerate(kfold.split(X_all_np, y_event_all), start=1):
        logger.info(f"\n{'='*60}\nFold {fold}/{n_folds} | CoxPH | IBS | {dataset_name}\n{'='*60}")
        
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
        best_val_mae_po = float("inf")
        best_model = None
        
        t_hp_start = time.perf_counter()
        for ci, params in enumerate(combos):
            logger.info(f" Fold {fold} HP {ci+1}/{len(combos)}: {params}")
            
            try:
                # Build and train CoxPH model via survhive adapter
                coxph = sv.CoxPH(rng_seed=seed + fold, **params)
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('error', category=RuntimeWarning)
                    try:
                        coxph.fit(X_train, y_train)
                    except (RuntimeWarning, ValueError, np.linalg.LinAlgError) as fit_err:
                        logger.warning(f"Fold {fold} HP {ci+1}: fit failed with {type(fit_err).__name__}: {fit_err}")
                        failed_combinations.append({
                            "fold": fold,
                            "hp_index": ci,
                            "params": params,
                            "reason": f"fit failed: {str(fit_err)}",
                            "error_type": type(fit_err).__name__
                        })
                        continue
                
                # Predict on val: get native predictions, interpolate once to eval grid
                try:
                    native_times_val, native_surv_val = get_native_surv_from_sksurv(coxph, X_val)
                    estimate_val = interpolate_survival_to_grid(native_times_val, native_surv_val, time_grid)
                    estimate_val = np.asarray(estimate_val, dtype=float)
                except (RuntimeWarning, ValueError, np.linalg.LinAlgError) as pred_err:
                    logger.warning(f"Fold {fold} HP {ci+1}: prediction failed with {type(pred_err).__name__}: {pred_err}")
                    failed_combinations.append({
                        "fold": fold,
                        "hp_index": ci,
                        "params": params,
                        "reason": f"prediction failed: {str(pred_err)}",
                        "error_type": type(pred_err).__name__
                    })
                    continue
                
                # Check if predictions are finite
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
                
                # Check for NaN or Inf
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
                
                if mae_po < best_val_mae_po:
                    best_val_mae_po = mae_po
                    best_hp_idx = ci
                    best_model = coxph
                    
            except Exception as e:
                logger.warning(f"Fold {fold} HP {ci+1} failed: {e}")
                failed_combinations.append({
                    "fold": fold,
                    "hp_index": ci,
                    "params": params,
                    "reason": str(e),
                    "error_type": type(e).__name__
                })
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
        
        best_params = combos[best_hp_idx]
        logger.info(f"Fold {fold} best HP: {best_params}, val_MAE_PO={best_val_mae_po:.6f}")
        
        # Refit model with best params on full training data (train+val)
        del best_model
        gc.collect()
        t_refit_start = time.perf_counter()
        X_trainfull = pd.concat([X_train, X_val])
        y_trainfull = np.concatenate([y_train, y_val])
        Y_trainfull, D_trainfull = split_y(y_trainfull)
        best_model = sv.CoxPH(rng_seed=seed + fold, **best_params)
        best_model.fit(X_trainfull, y_trainfull)
        logger.info(f"Fold {fold}: Refitted model with best HP on train+val data")
        t_refit_end = time.perf_counter()
        fold_refit_seconds = t_refit_end - t_refit_start
        
        # Plot feature importance (top 10 coefficients by absolute magnitude)
        coefs = pd.DataFrame(best_model.model_.coef_, index=X_trainfull.columns, columns=["coefficient"])
        
        # Store coefficients and hazard ratios for this fold
        coef_series = pd.Series(best_model.model_.coef_, index=X_trainfull.columns, name="beta")
        hazard_ratios = np.exp(coef_series)
        
        # Get top 10 by absolute magnitude
        top_10_idx = coef_series.abs().nlargest(10).index
        coef_series_top10 = coef_series[top_10_idx]
        hazard_ratios_top10 = hazard_ratios[top_10_idx]
        
        numeric_cols = [c for c in coefs.index if X_trainfull[c].nunique() > 2]
        coefs_numeric = coefs.loc[numeric_cols]
        n_excluded = len(coefs) - len(coefs_numeric)
        if n_excluded > 0:
            logger.info(f"Fold {fold}: excluded {n_excluded} binary/categorical features from importance plot")
        
        k = min(10, len(coefs_numeric))
        if save_plots and k > 0:
            top_k_order = coefs_numeric["coefficient"].abs().sort_values().tail(k).index
            
            fig, ax = plt.subplots(figsize=(6, 8))
            coefs_numeric.loc[top_k_order].plot.barh(ax=ax, legend=False)
            ax.set_xlabel("coefficient")
            ax.set_title(f"Top {k} Feature Importance (numerical only) - {dataset_name} - Fold {fold}")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            feat_imp_dir = plots_base / 'feature_importance' / dataset_name
            os.makedirs(feat_imp_dir, exist_ok=True)
            ts_plot = datetime.now().strftime("%Y%m%d_%H%M%S")
            feat_imp_path = feat_imp_dir / f"feat_imp_{dataset_name}_fold{fold}_{ts_plot}.png"
            plt.savefig(feat_imp_path, dpi=150, bbox_inches='tight')
            plt.close('all')
            logger.info(f"Saved feature importance plot: {feat_imp_path}")
        
        # Evaluate on test with best model
        time_grid_test = get_time_grid(df, y_trainfull, y_test)
        time_grid_test = subsample_time_grid(time_grid_test, ibs_grid_size, logger)

        # Baselines for reference
        n_test = len(Y_test)
        t_test = time_grid_test.astype(float)

        # random (const. prob of 0.5 everywhere)
        random_surv = 0.5 * np.ones((n_test, len(t_test)), dtype=np.float32)

        # Kaplan–Meier baseline
        km_t, km_s = kaplan_meier_estimator(D_trainfull.astype(bool), Y_trainfull.astype(float))
        km_func = StepFunction(km_t, km_s)
        km_curve_on_grid = km_func(t_test)
        km_surv = np.tile(km_curve_on_grid, (n_test, 1)).astype(np.float32)

        ev_random = SurvivalEvaluator(
            pred_survs=random_surv,
            time_coordinates=t_test,
            event_times=Y_test,
            event_indicators=D_test,
            train_event_times=Y_trainfull,
            train_event_indicators=D_trainfull,
        )
        random_ibs = float(ev_random.integrated_brier_score(
            num_points=None,
            target_times=t_test,
            IPCW_weighted=True,
            integration_method="trapz",
            draw_figure=False,
        ))
        random_mae_po = ev_random.mae(method="Pseudo_obs", weighted=True)

        ev_km = SurvivalEvaluator(
            pred_survs=km_surv,
            time_coordinates=t_test,
            event_times=Y_test,
            event_indicators=D_test,
            train_event_times=Y_trainfull,
            train_event_indicators=D_trainfull,
        )
        km_ibs = float(ev_km.integrated_brier_score(
            num_points=None,
            target_times=t_test,
            IPCW_weighted=True,
            integration_method="trapz",
            draw_figure=False,
        ))
        km_mae_po = ev_km.mae(method="Pseudo_obs", weighted=True)

        logger.info(
            f"Fold {fold} BASELINES: Random_IBS={random_ibs:.6f}, KM(train)_IBS={km_ibs:.6f}"
        )
        del ev_random, random_surv, ev_km, km_surv
        gc.collect()

        # Test evaluation: get native predictions, interpolate ONCE to eval grid
        t_infer_start = time.perf_counter()
        native_times_test, native_surv_test = get_native_surv_from_sksurv(best_model, X_test)
        surv_on_eval_grid = interpolate_survival_to_grid(
            native_times_test, native_surv_test, time_grid_test,
        )
        t_infer_end = time.perf_counter()
        fold_inference_seconds = t_infer_end - t_infer_start
        logger.info(f"Fold {fold} TIMING: HP search={fold_hp_search_seconds:.2f}s ({fold_hp_search_seconds/len(combos):.3f}s/combo), refit/model_load={fold_refit_seconds:.2f}s, inference={fold_inference_seconds:.4f}s")

        # IBS: pass interpolated survival + eval grid
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
        mae_po_test = ev_test.mae(method="Pseudo_obs", weighted=True)

        # C-index: pass same interpolated survival to pycox EvalSurv
        surv_df_test = pd.DataFrame(surv_on_eval_grid.T, index=time_grid_test)
        ev_pycox = EvalSurv(surv_df_test, Y_test.astype(float), D_test.astype(float), steps='post')
        test_c = float(ev_pycox.concordance_td(method='adj_antolini'))
        
        p_value, bin_stats = ev_test.d_calibration(return_details=True)
        x_cal_test = float(bin_stats["x_calibration"])
        
        if save_plots:
            calib_path = save_calibration_histogram(
                bin_stats, dataset_name, "CoxPH", fold,
                save_dir=plots_base / 'calibration_histogram',
                metric_label="ibs",
            )
            logger.info(f"Saved calibration plot: {calib_path}")
            plt.close('all')

            risk_scores = best_model.predict(X_test)
            km_path = plot_risk_stratified_km(
                risk_scores, Y_test, D_test, Y_train, D_train,
                dataset_name, "CoxPH", fold,
                save_dir=plots_base / 'km_curves',
                metric_label="ibs",
            )
            logger.info(f"Saved KM curve plot: {km_path}")
            plt.close('all')

            surv_path = plot_native_vs_interpolated(
                native_times_test, native_surv_test, time_grid_test, surv_on_eval_grid,
                test_idx, dataset_name, "CoxPH", fold, seed,
                save_dir=plots_base / 'individual_surv',
                metric_label="ibs",
                native_style="step",
            )
            logger.info(f"Saved native vs interpolated survival curves: {surv_path}")
            plt.close('all')
        
        logger.info(f"Fold {fold} TEST: IBS={test_ibs:.6f}, C-index={test_c:.6f}, x_cal={x_cal_test:.6f}, p={p_value:.6g}")
        
        fold_results.append({
            "fold": fold,
            "best_params": best_params,
            "fold_hp_search_seconds": fold_hp_search_seconds,
            "fold_hp_time_per_combo": fold_hp_search_seconds / len(combos),
            "fold_refit_seconds": fold_refit_seconds,
            "fold_inference_seconds": fold_inference_seconds,
            "val_mae_po": best_val_mae_po,
            "mae_po": float(mae_po_test),
            "km_train_mae_po": float(km_mae_po),
            "random_mae_po": float(random_mae_po),
            "test_ibs": test_ibs,
            "test_c": test_c,
            "test_x_cal": x_cal_test,
            "test_p": float(p_value),
            "random_ibs": random_ibs,
            "km_train_ibs": km_ibs,
            "n_dropped_val": n_dropped_val,
            "n_dropped_test": n_dropped_test,
            "coefficients_top10": coef_series_top10.to_dict(),
            "hazard_ratios_top10": hazard_ratios_top10.to_dict(),
        })
        
        del native_times_test, native_surv_test, surv_on_eval_grid
        del surv_df_test, best_model
        gc.collect()
    
    elapsed = time.perf_counter() - start_time
    
    """
    # Create split visualization
    plot_path = plot_cv_splits(
        fold_indices, len(X_all_np), n_folds,
        dataset_name, "CoxPH",
        save_dir=plots_base / 'cv_splits',
        metric_label="ibs",
    )
    logger.info(f"Saved split visualization: {plot_path}")
    """
    
    # Summary
    test_ibs_vals = [r["test_ibs"] for r in fold_results if np.isfinite(r["test_ibs"])]
    test_c_vals = [r["test_c"] for r in fold_results if np.isfinite(r["test_c"])]
    random_ibs_vals = [r["random_ibs"] for r in fold_results if np.isfinite(r["random_ibs"])]
    km_ibs_vals = [r["km_train_ibs"] for r in fold_results if np.isfinite(r["km_train_ibs"])]
    
    mean_test_ibs = float(np.mean(test_ibs_vals)) if test_ibs_vals else float("nan")
    std_test_ibs = float(np.std(test_ibs_vals)) if len(test_ibs_vals) > 1 else float("nan")
    mean_test_c = float(np.mean(test_c_vals)) if test_c_vals else float("nan")
    std_test_c = float(np.std(test_c_vals)) if len(test_c_vals) > 1 else float("nan")
    mean_random_ibs = float(np.mean(random_ibs_vals)) if random_ibs_vals else float("nan")
    mean_km_ibs = float(np.mean(km_ibs_vals)) if km_ibs_vals else float("nan")

    mae_po_vals = [r["mae_po"] for r in fold_results if np.isfinite(r["mae_po"])]
    mean_test_mae_po = float(np.mean(mae_po_vals)) if mae_po_vals else float("nan")
    std_test_mae_po = float(np.std(mae_po_vals)) if mae_po_vals else float("nan")

    km_mae_po_vals = [r["km_train_mae_po"] for r in fold_results if np.isfinite(r["km_train_mae_po"])]
    mean_km_mae_po = float(np.mean(km_mae_po_vals)) if km_mae_po_vals else float("nan")

    x_cal_vals = [r["test_x_cal"] for r in fold_results if np.isfinite(r["test_x_cal"])]
    mean_test_x_cal = float(np.mean(x_cal_vals)) if x_cal_vals else float("nan")
    std_test_x_cal = float(np.std(x_cal_vals)) if x_cal_vals else float("nan")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset: {dataset_name} | Time range: [{t_min:.3f}, {t_max:.3f}] {time_unit} | N={len(df)}")
    logger.info(f"K-FOLD CV SUMMARY ({n_folds} folds)")
    logger.info(f"Mean Test IBS: {mean_test_ibs:.6f} +/- {std_test_ibs:.6f}")
    logger.info(f"Mean Test C-index: {mean_test_c:.6f} +/- {std_test_c:.6f}")
    logger.info(f"Mean Random IBS: {mean_random_ibs:.6f}")
    logger.info(f"Mean KM(train) IBS: {mean_km_ibs:.6f}")
    if np.isfinite(mean_random_ibs) and mean_random_ibs > 0:
        logger.info(f"Improvement over Random: {(mean_random_ibs - mean_test_ibs) / mean_random_ibs * 100:.1f}%")
    if np.isfinite(mean_km_ibs) and mean_km_ibs > 0:
        logger.info(f"Improvement over KM: {(mean_km_ibs - mean_test_ibs) / mean_km_ibs * 100:.1f}%")
    logger.info(f"Mean Test MAE-PO: {mean_test_mae_po:.6f} +/- {std_test_mae_po:.6f}")
    logger.info(f"Mean KM(train) MAE-PO: {mean_km_mae_po:.6f}")
    logger.info(f"Mean Test X-Cal: {mean_test_x_cal:.6f} +/- {std_test_x_cal:.6f}")
    if np.isfinite(mean_km_mae_po) and mean_km_mae_po > 0:
        logger.info(f"MAE-PO Improvement over KM: {(mean_km_mae_po - mean_test_mae_po) / mean_km_mae_po * 100:.1f}%")
    logger.info(f"Total time: {elapsed:.2f}s")
    logger.info(f"{'='*60}")
    
    # Save results
    os.makedirs(results_dir, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = results_dir / f"kfold_coxph_{dataset_name}_{ts}_kfold.json"

    details = {
        "dataset": dataset_name,
        "t_min": t_min,
        "t_max": t_max,
        "time_unit": time_unit,
        "seed": seed,
        "n_folds": n_folds,
        "cindex_method": "pycox_adj_antolini",
        "interpolation": "linear",
        "optimization_metric": "none",
        "note": "CoxPH has no HPs to tune; all metrics are fold means over a fixed model.",
        "reported_metrics": ["antolini_cindex", "ibs", "mae_po", "xcal"],
        "mean_test_ibs": mean_test_ibs,
        "std_test_ibs": std_test_ibs,
        "mean_test_c": mean_test_c,
        "std_test_c": std_test_c,
        "mean_random_ibs": mean_random_ibs,
        "mean_km_ibs": mean_km_ibs,
        "mean_test_mae_po": mean_test_mae_po,
        "std_test_mae_po": std_test_mae_po,
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
        "failed_folds": failed_folds,
        "timestamp": ts,
        "budget_exceeded": budget_exceeded,
        "fold_time_budget_h": fold_time_budget_h,
        "ibs_grid_size": ibs_grid_size,
    }
    
    with open(details_path, "w") as f:
        json.dump(details, f, indent=2, default=str, separators=(',', ': '))
    
    logger.info(f"Saved results to: {details_path}")
    
    return details_path, details


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', 
                        #default=["veteran"],
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
        
        logs_dir = EXPERIMENT_RESULTS / 'logs' / 'coxph'
        os.makedirs(logs_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(logs_dir / f"coxph_kfold_ibs_{dataset_name}.log", mode="w"),
                logging.StreamHandler()
            ],
            force=True
        )
        
        try:
            optimization_coxph_kfold(
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
