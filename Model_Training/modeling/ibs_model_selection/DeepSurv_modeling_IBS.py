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
from copy import deepcopy
import numpy as np
import pandas as pd
from torch import nn
import argparse
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import StratifiedKFold

# survival specific imports
from pycox.models import CoxPH
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
    interpolate_survival_to_grid, apply_containment_fix,
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


#def _cuda_batch_iter(X, Y, D, batch_size: int, shuffle: bool):
#    """CUDA-optimized batch iterator."""
#    n = X.shape[0]
#    if shuffle:
#        idx = torch.randperm(n, device=X.device)
#    else:
#        idx = torch.arange(n, device=X.device)
#    for start in range(0, n, batch_size):
#        b = idx[start:start+batch_size]
#        yield X[b], Y[b], D[b]


def optimization_deepsurv_kfold(
    dataset_name: str,
    seed: int,
    device: str = "cpu",
    n_folds: int = 5,
    TIME_COL: str = "time",
    EVENT_COL: str = "event",
    ibs_every: int = 10,
    patience: int = 10,
    fold_time_budget_h: float = 3.0,
    ibs_grid_size: int = None,
    save_plots: bool = False,
):
    """
    Simple K-fold CV for DeepSurv:
    - Split data into k folds using StratifiedKFold (ensures each sample in test exactly once)
    - Per fold: 80% train_full, 20% test
    - Split train_full into 80% train, 20% val
    - Final split per fold: 64% train, 16% val, 20% test
    - Early stopping after `patience` evals without improvement (None = disabled)
    - Find best HP combo on val (lowest IBS + best epoch)
    - Evaluate that model on test
    - Average performance over k test sets
    """
    
    if device not in ['cpu', 'cuda'] and not device.startswith('cuda:'):
        raise ValueError(f"Invalid device: {device}")
    device = torch.device(device)
    
    logger.info(f"K-Fold CV DeepSurv IBS Script started (k={n_folds})")
    df = load_processed_dataset(dataset_name, method_name = "deepsurv", fix_split=False)
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
    
    # Pre-load to GPU if available
    if device.type == 'cuda':
        X_all_gpu = torch.tensor(X_all_np, dtype=torch.float32, device=device)
        y_time_gpu = torch.tensor(y_all["time"].copy(), dtype=torch.float32, device=device)
        y_event_gpu = torch.tensor(y_all["event"].copy().astype(int), dtype=torch.int64, device=device)
        logger.info(f"[GPU] Pre-loaded dataset to GPU")
    else:
        X_all_gpu = y_time_gpu = y_event_gpu = None

    logger.info(f"Dataset {dataset_name}: n={len(df)}, events={int(y_event_all.sum())}, censor_rate={1.0 - float(y_event_all.mean()):.3f}")

    # Load HP grid
    grid_dict, combos = load_hp_grid(repo_root, 'deepsurv', logger)

    results_dir = EXPERIMENT_RESULTS / 'results' / 'deepsurv' / 'ibs'
    plots_base = EXPERIMENT_RESULTS / 'plots' / 'deepsurv' / 'ibs'

    start_time = time.perf_counter()
    # Use StratifiedKFold: each sample appears exactly once as test across all folds
    # n_folds=5 means each fold has 20% test, 80% train_full
    # We then split train_full 80/20 to get train/val
    # Final split per fold: 64% train, 16% val, 20% test
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_results = []
    fold_indices = {}  # Store indices during training for reproducibility
    failed_combinations = []  # Track failed HP combinations
    failed_folds = []
    training_histories = []  # Store training curves for all fold/HP combinations
    
    FOLD_TIME_BUDGET_S = fold_time_budget_h * 3600
    budget_exceeded = False
    for fold, (trainfull_idx, test_idx) in enumerate(kfold.split(X_all_np, y_event_all), start=1):
        logger.info(f"\n{'='*60}\nFold {fold}/{n_folds} | DeepSurv | IBS | {dataset_name}\n{'='*60}")
        
        # Split trainfull into train and val 
        trainfull_idx = np.asarray(trainfull_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        
        # Further split trainfull into train/val (80/20 split)
        train_rel_idx, val_rel_idx = train_test_split(
            np.arange(len(trainfull_idx)), 
            test_size=0.2, 
            random_state=seed+fold,
            stratify=y_event_all[trainfull_idx]
        )
        train_idx = trainfull_idx[train_rel_idx]
        val_idx = trainfull_idx[val_rel_idx]
        
        # Store indices for reproducibility and visualization
        fold_indices[f"fold_{fold}"] = {
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "test_indices": test_idx.tolist()
        }
        
        X_train, y_train = X_all_np[train_idx], y_all[train_idx]
        X_val, y_val = X_all_np[val_idx], y_all[val_idx]
        X_test, y_test = X_all_np[test_idx], y_all[test_idx]
        
        # Containment fix: drop val/test samples with time >= train max
        val_mask, n_dropped_val, t_max_train = apply_containment_fix(y_train, y_val)
        if n_dropped_val > 0:
            logger.info(f"[Fold {fold}] Containment: dropped {n_dropped_val}/{len(y_val)} val samples (time >= {t_max_train})")
            y_val = y_val[val_mask]
            X_val = X_val[val_mask]
            val_idx = val_idx[val_mask]

        test_mask, n_dropped_test, _ = apply_containment_fix(y_train, y_test)
        if n_dropped_test > 0:
            logger.info(f"[Fold {fold}] Containment: dropped {n_dropped_test}/{len(y_test)} test samples (time >= {t_max_train})")
            y_test = y_test[test_mask]
            X_test = X_test[test_mask]
            test_idx = test_idx[test_mask]
        
        Y_train, D_train = split_y(y_train)
        Y_val, D_val = split_y(y_val)
        Y_test, D_test = split_y(y_test)
        
        logger.info(f"Train: {len(train_idx)} ({len(train_idx)/len(X_all_np)*100:.1f}%), Val: {len(val_idx)} ({len(val_idx)/len(X_all_np)*100:.1f}%), Test: {len(test_idx)} ({len(test_idx)/len(X_all_np)*100:.1f}%)")
        
        # GPU tensors
        if X_all_gpu is not None:
            X_train_t = X_all_gpu[train_idx]
            Y_train_t = y_time_gpu[train_idx]
            D_train_t = y_event_gpu[train_idx]
            X_val_t = X_all_gpu[val_idx]
            Y_val_t = y_time_gpu[val_idx]
            D_val_t = y_event_gpu[val_idx]
        else:
            X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
            Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
            D_train_t = torch.tensor(D_train, dtype=torch.int64, device=device)
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
            Y_val_t = torch.tensor(Y_val, dtype=torch.float32, device=device)
            D_val_t = torch.tensor(D_val, dtype=torch.int64, device=device)
        train_data = TensorDataset(X_train_t, Y_train_t, D_train_t)
        
        time_grid = get_time_grid(df, y_train, y_val)
        time_grid = subsample_time_grid(time_grid, ibs_grid_size, logger)
        if len(time_grid) == 0:
            logger.warning(f"Fold {fold}: empty time grid, skipping")
            continue
        
        # HP search
        best_hp_idx = None
        best_val_ibs = float("inf")
        best_epoch_for_best_hp = None
        best_state_for_best_hp = None
        
        t_hp_start = time.perf_counter()
        for ci, params in enumerate(combos):
            logger.info(f" Fold {fold} HP {ci+1}/{len(combos)}: {params}")
            
            hidden = int(params.get("hidden", 8))
            lr = float(params.get("learning_rate", params.get("lr", 1e-3)))
            epochs = int(params.get("epochs", 100))
            batch_size = int(params.get("batch_size", 128))
            dropout = float(params.get("dropout", 0.0))
            
            torch.manual_seed(seed + fold)
            np.random.seed(seed + fold)
            
            # Build model
            layers = [nn.Linear(X_train.shape[1], hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(hidden, 1, bias=False))
            net = nn.Sequential(*layers).to(device)
            model = CoxPH(net, device=device)
            loss_fn = model.loss
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            
            best_ibs_this_hp = float("inf")
            best_epoch_this_hp = None
            best_state_this_hp = None
            no_improve_count = 0
            
            # Track training history
            history = {
                "fold": fold,
                "hp_index": ci,
                "params": params,
                "dataset": dataset_name,
                "epochs": [],
                "train_loss": [],
                "val_loss": [],
                "val_ibs": []
            }
            
            for epoch in range(1, epochs + 1):
                # Train
                net.train()
                epoch_train_losses = []
                train_loader = DataLoader(train_data, batch_size, shuffle=True)
                for xb, yb, db in train_loader:
                    out = net(xb)
                    loss = loss_fn(out, yb, db)
                    if torch.isfinite(loss):
                        epoch_train_losses.append(loss.item())
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                        opt.step()
                
                # Compute validation loss
                net.eval()
                epoch_val_losses = []
                with torch.inference_mode():
                    val_loader = DataLoader(TensorDataset(X_val_t, Y_val_t, D_val_t), batch_size, shuffle=False)
                    for xb, yb, db in val_loader:
                        out = net(xb)
                        loss = loss_fn(out, yb, db)
                        if torch.isfinite(loss):
                            epoch_val_losses.append(loss.item())
                
                # Store losses
                if epoch_train_losses:
                    history["epochs"].append(epoch)
                    history["train_loss"].append(np.mean(epoch_train_losses))
                    history["val_loss"].append(np.mean(epoch_val_losses) if epoch_val_losses else np.nan)
                
                # Evaluate IBS on val every ibs_every epochs
                if epoch % ibs_every == 0:
                    net.eval()
                    with torch.inference_mode():
                        try:
                            model.compute_baseline_hazards(input=X_train, target=(Y_train, D_train))
                            surv_val_df = model.predict_surv_df(X_val.astype("float32"), batch_size=batch_size)
                            native_times_v = surv_val_df.index.values
                            native_surv_v = surv_val_df.values.T
                            surv_val_interp = interpolate_survival_to_grid(native_times_v, native_surv_v, time_grid)
                            surv_val_interp = np.clip(surv_val_interp, 1e-6, 1.0 - 1e-6)
                            
                            ev = SurvivalEvaluator(
                                pred_survs=surv_val_interp,
                                time_coordinates=time_grid.astype(float),
                                event_times=Y_val,
                                event_indicators=D_val,
                                train_event_times=Y_train,
                                train_event_indicators=D_train,
                            )
                            ibs = float(ev.integrated_brier_score(
                                num_points=None,
                                target_times=time_grid.astype(float),
                                IPCW_weighted=True,
                                integration_method="trapz",
                                draw_figure=False,
                            ))
                            
                            # Check for NaN or Inf
                            if not np.isfinite(ibs):
                                reason = "NaN" if np.isnan(ibs) else "Inf"
                                logger.warning(f"Fold {fold} HP {ci+1} Epoch {epoch}: val_IBS is {reason}, skipping")
                                history["val_ibs"].append(np.nan)
                                no_improve_count += 1
                            else:
                                logger.info(f"Fold {fold} HP {ci+1}/{len(combos)} Epoch {epoch}: val_IBS={ibs:.6f}")
                                history["val_ibs"].append(ibs)
                                
                                if ibs < best_ibs_this_hp:
                                    best_ibs_this_hp = ibs
                                    best_epoch_this_hp = epoch
                                    best_state_this_hp = deepcopy(net.state_dict())
                                    no_improve_count = 0
                                else:
                                    no_improve_count += 1
                        except Exception as e:
                            logger.warning(f"Fold {fold} HP {ci+1} Epoch {epoch} failed: {e}")
                            history["val_ibs"].append(np.nan)
                            no_improve_count += 1
                else:
                    history["val_ibs"].append(np.nan)
                
                # Early stopping check
                if patience is not None and no_improve_count >= patience:
                    logger.info(
                        f"Fold {fold} HP {ci+1}: early stopping at "
                        f"epoch {epoch} (no improvement for "
                        f"{patience} evals)"
                    )
                    break
            
            # Store history
            training_histories.append(history)
            
            # Check if this HP combo is best
            if not np.isfinite(best_ibs_this_hp):
                reason = "NaN" if np.isnan(best_ibs_this_hp) else "Inf"
                logger.warning(f"Fold {fold} HP {ci+1}: best val_IBS is {reason}, skipping")
                failed_combinations.append({
                    "fold": fold,
                    "hp_index": ci,
                    "params": params,
                    "reason": f"best val_IBS is {reason}",
                    "val_ibs": str(best_ibs_this_hp)
                })
            elif best_ibs_this_hp < best_val_ibs:
                best_val_ibs = best_ibs_this_hp
                best_hp_idx = ci
                best_epoch_for_best_hp = best_epoch_this_hp
                best_state_for_best_hp = best_state_this_hp
            if 'net' in locals(): del net
            if 'model' in locals(): del model
            if 'loss_fn' in locals(): del loss_fn
            if 'opt' in locals(): del opt
            gc.collect()
            torch.cuda.empty_cache()

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
        logger.info(f"Fold {fold} best HP: {best_params}, val_IBS={best_val_ibs:.6f}, epoch={best_epoch_for_best_hp}")
        
        try:
            # Evaluate on test with best model
            batch_size_best = int(best_params.get("batch_size", 128))
            
            t_refit_start = time.perf_counter()
            # Rebuild model with best params and load best state
            hidden = int(best_params.get("hidden", 8))
            dropout = float(best_params.get("dropout", 0.0))
            layers = [nn.Linear(X_train.shape[1], hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(hidden, 1, bias=False))
            net = nn.Sequential(*layers).to(device)
            net.load_state_dict(best_state_for_best_hp)
            net.eval()
            
            final_model = CoxPH(net, device=device)
            
            # Refit on train+val for final test evaluation
            X_trainfull = np.concatenate([X_train, X_val])
            y_trainfull = np.concatenate([y_train, y_val])
            Y_trainfull, D_trainfull = split_y(y_trainfull)
            
            final_model.compute_baseline_hazards(input=X_trainfull, target=(Y_trainfull, D_trainfull))
            t_refit_end = time.perf_counter()
            fold_refit_seconds = t_refit_end - t_refit_start
            
            time_grid_test = get_time_grid(df, y_trainfull, y_test)
            time_grid_test = subsample_time_grid(time_grid_test, ibs_grid_size, logger)
            
            # Baselines for reference
            n_test = len(Y_test)
            t_test = time_grid_test.astype(float)
            
            random_ibs, km_ibs, km_mae_po = compute_baselines(
                Y_test, D_test, Y_trainfull, D_trainfull, t_test, n_test, logger, fold
            )
            
            # Test evaluation
            t_infer_start = time.perf_counter()
            surv_test_df = final_model.predict_surv_df(X_test.astype("float32"), batch_size=batch_size_best)
            native_times_test = surv_test_df.index.values
            native_surv_test = surv_test_df.values.T
            surv_on_eval_grid = interpolate_survival_to_grid(
                native_times_test, native_surv_test, time_grid_test,
            )
            surv_on_eval_grid = np.clip(surv_on_eval_grid, 1e-6, 1.0 - 1e-6)
            t_infer_end = time.perf_counter()
            fold_inference_seconds = t_infer_end - t_infer_start
            
            test_ibs, test_c, x_cal_test, p_value, bin_stats, mae_po_score = evaluate_on_test(
                surv_on_eval_grid, time_grid_test, Y_test, D_test, Y_trainfull, D_trainfull
            )
            
            logger.info(f"Fold {fold} TIMING: HP search={fold_hp_search_seconds:.2f}s ({fold_hp_search_seconds/len(combos):.3f}s/combo), refit/model_load={fold_refit_seconds:.2f}s, inference={fold_inference_seconds:.4f}s")
            
            if save_plots:
                calib_path = save_calibration_histogram(bin_stats, dataset_name, "DeepSurv", fold, save_dir=plots_base / 'calibration_histogram' / 'deepsurv', metric_label="ibs")
                logger.info(f"Saved calibration plot: {calib_path}")
                
                with torch.inference_mode():
                    risk_scores = -final_model.predict(torch.tensor(X_test, dtype=torch.float32, device=device), batch_size=batch_size_best).cpu().numpy().flatten()
                km_path = plot_risk_stratified_km(risk_scores, Y_test, D_test, Y_train, D_train, dataset_name, "DeepSurv", fold, save_dir=plots_base / 'km_curves' / 'deepsurv', metric_label="ibs")
                logger.info(f"Saved KM curve plot: {km_path}")
                
                try:
                    surv_path = plot_native_vs_interpolated(
                        native_times_test, native_surv_test, time_grid_test, surv_on_eval_grid,
                        test_idx, dataset_name, "DeepSurv", fold, seed,
                        save_dir=plots_base / 'individual_surv' / 'deepsurv', metric_label="ibs", native_style="line"
                    )
                    logger.info(f"Saved individual survival curves: {surv_path}")
                except Exception as e:
                    logger.error(f"Failed to plot individual survival curves: {e}")
                    plt.close('all')
            
            logger.info(f"Fold {fold} TEST: IBS={test_ibs:.6f}, C-index={test_c:.6f}, x_cal={x_cal_test:.6f}, p={p_value:.6g}, MAE_PO={mae_po_score:.6f}")
            
            fold_results.append({
                "fold": fold,
                "best_hp_index": best_hp_idx,
                "best_params": best_params,
                "best_epoch": best_epoch_for_best_hp,
                "val_ibs": best_val_ibs,
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
                "fold_hp_search_seconds": fold_hp_search_seconds,
                "fold_hp_time_per_combo": fold_hp_search_seconds / len(combos),
                "fold_refit_seconds": fold_refit_seconds,
                "fold_inference_seconds": fold_inference_seconds,
            })
        except Exception as e:
            logger.error(f"Fold {fold} test evaluation failed: {e}")
            failed_folds.append({
                "fold": fold,
                "best_hp_index": best_hp_idx,
                "best_params": best_params,
                "best_val_ibs": best_val_ibs,
                "best_epoch": best_epoch_for_best_hp,
                "reason": repr(e),
            })
        finally:
            if 'best_state_for_best_hp' in locals() and best_state_for_best_hp is not None:
                del best_state_for_best_hp
            gc.collect()
            torch.cuda.empty_cache()
    
    elapsed = time.perf_counter() - start_time
    
    """
    plot_path = plot_cv_splits(fold_indices, len(X_all_np), n_folds, dataset_name, "DeepSurv", save_dir=plots_base / 'cv_splits', metric_label="ibs")
    logger.info(f"Saved split visualization: {plot_path}")
    """
    
    (mean_test_ibs, std_test_ibs, mean_test_c, std_test_c,
     mean_random_ibs, mean_km_ibs, mean_mae_po, std_mae_po, mean_km_mae_po) = log_cv_summary(
        fold_results, logger, dataset_name, t_min, t_max, time_unit, n_folds, elapsed, n=len(df)
    )
    
    # Save results
    os.makedirs(results_dir, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = results_dir / f"kfold_deepsurv_{dataset_name}_{ts}_ibs.json"
    
    details = {
        "dataset": dataset_name,
        "seed": seed,
        "n_folds": n_folds,
        "cindex_method": "pycox_adj_antolini",
        "interpolation": "linear",
        "optimization_metric": "IBS",
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
        "fold_indices": fold_indices if len(df) <= 10_000 else "omitted (n > 10k)",
        "failed_combinations": failed_combinations,
        "num_failed_combinations": len(failed_combinations),
        "failed_folds": failed_folds,
        "num_failed_folds": len(failed_folds),
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
    
    if save_plots:
        if len(training_histories) >= 9:
            rng = np.random.default_rng(seed)
            selected_indices = rng.choice(len(training_histories), size=9, replace=False)
        else:
            selected_indices = range(len(training_histories))
        
        if len(selected_indices) > 0:
            fig, axes = plt.subplots(3, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for idx, hist_idx in enumerate(selected_indices):
                if idx >= 9:
                    break
                
                hist = training_histories[hist_idx]
                ax = axes[idx]
                ax2 = ax.twinx()
                
                epochs = hist["epochs"]
                ax.plot(epochs, hist["train_loss"], 'b-', label='Train Loss', linewidth=1.5)
                ax.plot(epochs, hist["val_loss"], 'r-', label='Val Loss', linewidth=1.5)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss', color='k')
                ax.tick_params(axis='y', labelcolor='k')

                ibs_epochs = [e for e in epochs if not np.isnan(hist["val_ibs"][epochs.index(e)])]
                ibs_values = [hist["val_ibs"][epochs.index(e)] for e in ibs_epochs]
                if ibs_values:
                    ax2.plot(ibs_epochs, ibs_values, 'g--', label='Val IBS', linewidth=1.5, marker='o', markersize=3)
                    ax2.set_ylabel('Val IBS', color='g')
                    ax2.tick_params(axis='y', labelcolor='g')

                title = f"Fold {hist['fold']}, {hist['dataset']}\nHP {hist['hp_index']+1}: {hist['params']}"
                ax.set_title(title, fontsize=8)
                ax.legend(loc='upper left', fontsize=7)
                if ibs_values:
                    ax2.legend(loc='upper right', fontsize=7)
                ax.grid(True, alpha=0.3)
            
            for idx in range(len(selected_indices), 9):
                axes[idx].axis('off')
            
            plt.tight_layout()
            curves_dir = plots_base / 'training_curves' / 'deepsurv'
            os.makedirs(curves_dir, exist_ok=True)
            ts_curves = datetime.now().strftime("%Y%m%d_%H%M%S")
            curves_path = curves_dir / f"training_curves_{dataset_name}_deepsurv_ibs_{ts_curves}.png"
            plt.savefig(curves_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved training curves: {curves_path}")
    
    return details_path, details


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=["ovarian", "breast", "micro.censure", "e1684", "acath", "prostateSurvival", "NSBCD", "nwtco", "nki70", "Z243", "burn", "veteran", "divorce", "UnempDur", "Unemployment", "support2"])
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--ibs_every', type=int, default=1,
                        help='Evaluate val IBS every N epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10).')
    parser.add_argument('--fold_time_budget_h', type=float, default=3.0,
                        help='Per-fold time budget in hours (default: 3.0)')
    parser.add_argument('--ibs_grid_size', type=int, default=None,
                        help='Subsample IBS time grid to this many evenly-spaced points (default: None = no subsampling)')
    parser.add_argument('--save_plots', action='store_true', default=False)
    args = parser.parse_args()

    device = args.device
    logger.info(f"Using device: {device}")

    seed = 42
    
    for dataset_name in args.datasets:
        logger.info(f"\nProcessing {dataset_name}")
        
        logs_dir = EXPERIMENT_RESULTS / 'logs' / 'deepsurv' / 'ibs'
        os.makedirs(logs_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(logs_dir / f"deepsurv_kfold_ibs_{dataset_name}.log", mode="w"),
                logging.StreamHandler()
            ],
            force=True
        )
        
        try:
            optimization_deepsurv_kfold(
                dataset_name=dataset_name,
                seed=seed,
                device=device,
                n_folds=args.n_folds,
                ibs_every=args.ibs_every,
                patience=args.patience,
                fold_time_budget_h=args.fold_time_budget_h,
                ibs_grid_size=args.ibs_grid_size,
                save_plots=args.save_plots,
            )
        except Exception as e:
            logger.error(f"Failed: {e}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()
