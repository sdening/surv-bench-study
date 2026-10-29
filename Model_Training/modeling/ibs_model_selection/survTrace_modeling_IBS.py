import gc
import ctypes
import ctypes.util
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
import argparse
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from easydict import EasyDict

from SurvivalEVAL.Evaluator import SurvivalEvaluator

def get_repo_root():
    return Path(__file__).resolve().parent.parent.parent.parent

repo_root = get_repo_root()
EXPERIMENT_RESULTS = repo_root / 'Model_Training' / 'modeling' / 'experiment_results'
sys.path.append(str(repo_root / 'Data_Preprocessing' / 'src'))
sys.path.append(str(repo_root / 'Model_Training' / 'src'))

from _path_setup import use_local_survtrace
use_local_survtrace()

from survtrace.utils import LabelTransform
from survtrace.config import STConfig
from survtrace.model import SurvTraceSingle
from survtrace.losses import NLLPCHazardLoss
from survtrace.optim import BERTAdam
from modeling_utils import (
    get_time_grid, subsample_time_grid, split_y, load_processed_dataset,
    interpolate_survival_to_grid, apply_containment_fix,
    compute_baselines, evaluate_on_test, log_cv_summary,
    load_hp_grid,
)
from plot_utils import plot_native_vs_interpolated, plot_risk_stratified_km, save_calibration_histogram, plot_cv_splits

warnings.filterwarnings('ignore', category=UserWarning, module='SurvivalEVAL')

if not hasattr(pd.Series, "is_monotonic"):
    pd.Series.is_monotonic = property(lambda self: self.is_monotonic_increasing)


def _malloc_trim():
    lib = ctypes.util.find_library("c")
    if lib:
        try: ctypes.CDLL(lib).malloc_trim(0)
        except Exception: pass

def _fix_mmap_threshold():
    lib = ctypes.util.find_library("c")
    if lib:
        try:
            libc = ctypes.CDLL(lib)
            M_MMAP_THRESHOLD = -3
            libc.mallopt(M_MMAP_THRESHOLD, 8 * 1024 * 1024)
        except Exception: pass

_fix_mmap_threshold()


def compute_val_ibs(model, df_X, Y, D, Y_train, D_train, time_grid, batch_size=128):
    """Compute Integrated Brier Score using standardized linear interpolation."""
    surv_df = model.predict_surv_df(df_X, batch_size=batch_size)
    native_times = surv_df.index.values.astype(float)
    native_surv = surv_df.values.T  # (n_patients, n_times)
    surv_interp = interpolate_survival_to_grid(native_times, native_surv, time_grid)
    surv_interp = np.clip(surv_interp, 1e-6, 1.0 - 1e-6)

    ev = SurvivalEvaluator(
        pred_survs=surv_interp,
        time_coordinates=time_grid.astype(float),
        event_times=Y,
        event_indicators=D,
        train_event_times=Y_train,
        train_event_indicators=D_train,
    )
    return float(ev.integrated_brier_score(
        num_points=None,
        target_times=time_grid.astype(float),
        IPCW_weighted=True,
        integration_method="trapz",
        draw_figure=False,
    ))


def optimization_survtrace_kfold(
    dataset_name: str,
    seed: int,
    device: str = "cpu",
    n_folds: int = 5,
    TIME_COL: str = "time",
    EVENT_COL: str = "event",
    ibs_every: int = 10,
    patience: int = 10,
    epoch_metric: str = "val_ibs",
    fold_time_budget_h: float = 3.0,
    trim_memory: str = "epoch",
    ibs_grid_size: int = None,
    save_plots: bool = False,
):
    """K-fold CV for SurvTrace with IBS based HP selection
    """
    assert epoch_metric in ("val_ibs", "val_loss"), \
        f"epoch_metric must be 'val_ibs' or 'val_loss', got '{epoch_metric}'"
    assert trim_memory in ("epoch", "hp", "fold", "none"), \
        f"trim_memory must be one of epoch/hp/fold/none, got '{trim_memory}'"
    _trim_at_epoch = trim_memory == "epoch"
    _trim_at_hp = trim_memory in ("epoch", "hp")
    _trim_at_fold = trim_memory in ("epoch", "hp", "fold")

    logger.info(f"Epoch selection metric: {epoch_metric}")
    logger.info(f"Memory trim level: {trim_memory}")
    if device not in ['cpu', 'cuda'] and not device.startswith('cuda:'):
        raise ValueError(f"Invalid device: {device}")
    device = torch.device(device)

    logger.info(f"K-Fold CV SurvTrace IBS started (k={n_folds})")
    df = load_processed_dataset(
        dataset_name, method_name="survtrace_DL", fix_split=False,
    )
    t_min = float(df[TIME_COL].min())
    t_max = float(df[TIME_COL].max())
    _cfg_path = repo_root / 'Data_Preprocessing' / 'configs' / f'{dataset_name}.yaml'
    if not _cfg_path.exists():
        print(f"[WARNING] Dataset config not found: {_cfg_path} — TIME_UNIT defaulting to 'unknown'")
    time_unit = yaml.safe_load(_cfg_path.read_text()).get('TIME_UNIT', 'unknown') if _cfg_path.exists() else 'unknown'

    X_all_df = df.drop(columns=[EVENT_COL, TIME_COL])
    X_all_np = X_all_df.to_numpy().astype("float32", copy=False)
    n_features = X_all_np.shape[1]
    y_all = np.array(
        list(zip(df[EVENT_COL].astype(bool), df[TIME_COL].astype(float))),
        dtype=[("event", "?"), ("time", "<f8")],
    )
    y_event_all = y_all["event"].astype(int)

    logger.info(
        f"Dataset {dataset_name}: n={len(df)}, features={n_features}, "
        f"events={int(y_event_all.sum())}, censor_rate={1.0 - float(y_event_all.mean()):.3f}"
    )

    MAX_FEATURES = 1000
    if n_features > MAX_FEATURES:
        logger.warning(
            f"SKIPPING {dataset_name}: {n_features} features exceeds "
            f"SurvTrace limit of {MAX_FEATURES} (memory issues with "
            f"attention-based architecture). Returning None."
        )
        return None, None

    # Load HP grid
    grid_dict, combos = load_hp_grid(repo_root, 'survtrace_DL', logger)

    results_dir = EXPERIMENT_RESULTS / 'results' / 'survtrace' / 'ibs'
    plots_base = EXPERIMENT_RESULTS / 'plots' / 'survtrace' / 'ibs'

    start_time = time.perf_counter()
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []
    fold_indices = {}
    failed_combinations = []
    failed_folds = []
    training_histories = []

    FOLD_TIME_BUDGET_S = fold_time_budget_h * 3600
    budget_exceeded = False
    for fold, (trainfull_idx, test_idx) in enumerate(kfold.split(X_all_np, y_event_all), start=1):
        logger.info(f"\n{'='*60}\nFold {fold}/{n_folds} | SurvTrace | IBS | {dataset_name}\n{'='*60}")

        trainfull_idx = np.asarray(trainfull_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)

        train_rel_idx, val_rel_idx = train_test_split(
            np.arange(len(trainfull_idx)),
            test_size=0.2,
            random_state=seed + fold,
            stratify=y_event_all[trainfull_idx],
        )
        train_idx = trainfull_idx[train_rel_idx]
        val_idx = trainfull_idx[val_rel_idx]

        fold_indices[f"fold_{fold}"] = {
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "test_indices": test_idx.tolist(),
        }

        X_train, y_train = X_all_np[train_idx], y_all[train_idx]
        X_val, y_val = X_all_np[val_idx], y_all[val_idx]
        X_test, y_test = X_all_np[test_idx], y_all[test_idx]

        # Containment fix: drop samples with time >= max train time
        val_mask, n_dropped_val, t_max_train = apply_containment_fix(y_train, y_val)
        if n_dropped_val > 0:
            logger.info(f"[Fold {fold}] Containment: dropped {n_dropped_val}/{len(y_val)} val samples (time >= {t_max_train})")
            y_val = y_val[val_mask]
            X_val = X_val[val_mask]

        test_mask, n_dropped_test, _ = apply_containment_fix(y_train, y_test)
        if n_dropped_test > 0:
            logger.info(f"[Fold {fold}] Containment: dropped {n_dropped_test}/{len(y_test)} test samples (time >= {t_max_train})")
            y_test = y_test[test_mask]
            X_test = X_test[test_mask]
            test_idx = test_idx[test_mask]

        Y_train, D_train = split_y(y_train)
        Y_val, D_val = split_y(y_val)
        Y_test, D_test = split_y(y_test)

        logger.info(
            f"Train: {len(X_train)} ({len(X_train)/len(X_all_np)*100:.1f}%), "
            f"Val: {len(X_val)} ({len(X_val)/len(X_all_np)*100:.1f}%), "
            f"Test: {len(X_test)} ({len(X_test)/len(X_all_np)*100:.1f}%)"
        )

        time_grid = get_time_grid(df, y_train, y_val)
        time_grid = subsample_time_grid(time_grid, ibs_grid_size, logger)
        if time_grid is None or len(time_grid) == 0:
            logger.warning(f"Fold {fold}: empty time grid, skipping")
            continue

        # DataFrames for predict_surv_df
        df_val_X = pd.DataFrame(X_val.astype("float32"))
        df_test_X = pd.DataFrame(X_test.astype("float32"))

        #HP search
        best_hp_idx = None
        best_val_ibs = float("inf")
        best_epoch_for_best_hp = None
        best_state_for_best_hp = None
        best_cfg_for_best_hp = None
        best_labtrans_for_best_hp = None

        t_hp_start = time.perf_counter()
        for ci, params in enumerate(combos):
            logger.info(f"  Fold {fold} HP {ci+1}/{len(combos)}: {params}")

            torch.manual_seed(seed + fold)
            np.random.seed(seed + fold)

            # -- Discretization --
            event_mask = D_train.astype(bool)
            event_times = Y_train[event_mask].astype(float)
            horizons = np.array([0.25, 0.5, 0.75])
            cuts = np.array(
                [Y_train.astype(float).min()]
                + np.quantile(event_times, horizons).tolist()
                + [Y_train.astype(float).max()]
            )
            cuts = np.unique(cuts)

            labtrans = LabelTransform(cuts=cuts)
            labtrans.fit(Y_train, D_train)

            idx_train, ev_train, frac_train = labtrans.transform(Y_train, D_train)
            idx_val, ev_val, frac_val = labtrans.transform(Y_val, D_val)

            # -- Build model config --
            hidden_factor = int(params["hidden_factor"])
            num_attention_heads = int(params["num_attention_heads"])

            cfg = EasyDict(deepcopy(dict(STConfig)))
            cfg["hidden_size"] = hidden_factor * num_attention_heads
            cfg["intermediate_size"] = int(params.get("intermediate_size", 64))
            cfg["num_hidden_layers"] = int(params.get("num_hidden_layers", 3))
            cfg["num_attention_heads"] = num_attention_heads
            cfg["hidden_dropout_prob"] = float(params.get("hidden_dropout", 0.0))
            cfg["attention_probs_dropout_prob"] = float(params.get("attention_dropout", 0.1))
            cfg["seed"] = seed + fold
            cfg["num_categorical_feature"] = 0
            cfg["num_numerical_feature"] = n_features
            cfg["num_feature"] = n_features
            cfg["vocab_size"] = 0
            cfg["labtrans"] = labtrans
            cfg["duration_index"] = labtrans.cuts.astype(float)
            cfg["out_feature"] = int(labtrans.out_features)

            # -- Create model, loss, optimizer --
            model = SurvTraceSingle(cfg).to(device)
            if device.type == 'cuda':
                model.use_gpu = True
            loss_fn = NLLPCHazardLoss()

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            param_list = list(model.named_parameters())
            optimizer = BERTAdam(
                [
                    {'params': [p for n, p in param_list if not any(nd in n for nd in no_decay)]},
                    {'params': [p for n, p in param_list if any(nd in n for nd in no_decay)]},
                ],
                lr=1e-3,
                weight_decay_rate=0.0,
                max_grad_norm=1.0,
            )

            epochs = int(params["epochs"])
            batch_size = int(params["batch_size"])

            # -- Prepare training tensors --
            X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
            y_labels_t = torch.stack([
                torch.tensor(idx_train, dtype=torch.long),
                torch.tensor(ev_train.astype(np.int64), dtype=torch.long),
                torch.tensor(frac_train, dtype=torch.float32),
            ], dim=1).to(device)

            n_train = len(X_train_t)

            # Tracking: lower is better for both val_ibs and val_loss
            best_metric_this_hp = float("inf")
            best_ibs_at_best_epoch = float("inf")
            best_epoch_this_hp = None
            best_state_this_hp = None
            patience_counter = 0

            history = {
                "fold": fold, "hp_index": ci, "params": params,
                "dataset": dataset_name,
                "epochs": [], "train_loss": [], "val_loss": [], "val_ibs": [],
            }

            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
            x_cat_val = torch.zeros(len(X_val), 0, dtype=torch.long, device=device)
            y_val_t = torch.stack([
                torch.tensor(idx_val, dtype=torch.long),
                torch.tensor(ev_val.astype(np.int64), dtype=torch.long),
                torch.tensor(frac_val, dtype=torch.float32),
            ], dim=1).to(device)

            try:
                for epoch in range(1, epochs + 1):
                    #  TRAIN 
                    model.train()
                    perm = torch.randperm(n_train, device=device)
                    epoch_loss = 0.0
                    n_batches = 0

                    for start in range(0, n_train, batch_size):
                        idx = perm[start:start + batch_size]
                        x_num = X_train_t[idx]
                        x_cat = torch.zeros(len(idx), 0, dtype=torch.long, device=device)
                        yb = y_labels_t[idx]

                        phi = model(input_ids=x_cat, input_nums=x_num)
                        batch_loss = loss_fn(
                            phi[1],
                            yb[:, 0].long(),
                            yb[:, 1].long(),
                            yb[:, 2].float(),
                        )

                        if torch.isfinite(batch_loss):
                            optimizer.zero_grad()
                            batch_loss.backward()
                            optimizer.step()
                            epoch_loss += batch_loss.item()
                            n_batches += 1

                    avg_train_loss = epoch_loss / max(n_batches, 1)

                    # # VAL loss
                    model.eval()
                    with torch.inference_mode():
                        phi_val = model(input_ids=x_cat_val, input_nums=X_val_t)
                        val_loss = loss_fn(
                            phi_val[1], y_val_t[:, 0].long(),
                            y_val_t[:, 1].long(), y_val_t[:, 2].float(),
                        )
                        avg_val_loss = float(val_loss.item()) if torch.isfinite(val_loss) else np.nan

                    history["epochs"].append(epoch)
                    history["train_loss"].append(avg_train_loss)
                    history["val_loss"].append(avg_val_loss)

                    #checkpoointing on  val
                    if epoch_metric == "val_loss" and np.isfinite(avg_val_loss):
                        improved = avg_val_loss < best_metric_this_hp
                        if improved:
                            best_metric_this_hp = avg_val_loss
                            best_epoch_this_hp = epoch
                            best_state_this_hp = deepcopy(model.state_dict())
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if epoch % 10 == 0 or epoch == epochs:
                            logger.info(
                                f"    Epoch {epoch}/{epochs}: "
                                f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f} "
                                f"(best_loss={best_metric_this_hp:.4f}@ep{best_epoch_this_hp})"
                            )

                        if patience is not None and patience_counter >= patience:
                            logger.info(
                                f"    Early stop at epoch {epoch}: no val_loss improvement "
                                f"for {patience} epochs "
                                f"(best_loss={best_metric_this_hp:.4f}@ep{best_epoch_this_hp})"
                            )
                            break

                    # ibs on val
                    if epoch % ibs_every == 0:
                        model.eval()
                        with torch.inference_mode():
                            try:
                                ibs = compute_val_ibs(
                                    model, df_val_X, Y_val, D_val,
                                    Y_train, D_train, time_grid,
                                    batch_size=batch_size,
                                )

                                if not np.isfinite(ibs):
                                    reason = "NaN" if np.isnan(ibs) else "Inf"
                                    logger.warning(f"Fold {fold} HP {ci+1} Epoch {epoch}: val IBS is {reason}")
                                    history["val_ibs"].append(np.nan)
                                    continue

                                history["val_ibs"].append(ibs)

                                # checkpoint on val
                                if epoch_metric == "val_ibs":
                                    improved = ibs < best_metric_this_hp
                                    if improved:
                                        best_metric_this_hp = ibs
                                        best_ibs_at_best_epoch = ibs
                                        best_epoch_this_hp = epoch
                                        best_state_this_hp = deepcopy(model.state_dict())
                                        patience_counter = 0
                                    else:
                                        patience_counter += 1

                                    if patience is not None and patience_counter >= patience:
                                        logger.info(
                                            f"    Early stop at epoch {epoch}: no val_ibs improvement "
                                            f"for {patience} eval rounds "
                                            f"(best_IBS={best_metric_this_hp:.6f}@ep{best_epoch_this_hp})"
                                        )
                                        break

                                if epoch % 10 == 0 or epoch == epochs:
                                    logger.info(
                                        f"    Epoch {epoch}/{epochs}: "
                                        f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
                                        f"val_IBS={ibs:.6f} (best_{epoch_metric}={best_metric_this_hp:.6f}@ep{best_epoch_this_hp})"
                                    )

                            except Exception as e:
                                logger.warning(f"Fold {fold} HP {ci+1} Epoch {epoch} IBS failed: {e}")
                                history["val_ibs"].append(np.nan)
                    else:
                        history["val_ibs"].append(np.nan)

                    if _trim_at_epoch:
                        gc.collect()
                        _malloc_trim()

            except Exception as e:
                logger.exception(f"Fold {fold} HP {ci+1} training crashed: {e}")
                failed_combinations.append({
                    "fold": fold, "hp_index": ci, "params": params,
                    "reason": "exception", "exception": repr(e),
                })
                del model, optimizer, loss_fn, X_train_t, y_labels_t, X_val_t, x_cat_val, y_val_t
                del best_state_this_hp
                gc.collect()
                if _trim_at_hp: _malloc_trim()
                torch.cuda.empty_cache()
                continue

            training_histories.append(history)

            # -- For val_loss mode: compute IBS at the best-loss checkpoint --
            if epoch_metric == "val_loss" and best_state_this_hp is not None:
                model.load_state_dict(best_state_this_hp)
                model.eval()
                with torch.inference_mode():
                    try:
                        best_ibs_at_best_epoch = compute_val_ibs(
                            model, df_val_X, Y_val, D_val,
                            Y_train, D_train, time_grid,
                            batch_size=batch_size,
                        )
                    except Exception as e:
                        logger.warning(f"Fold {fold} HP {ci+1}: IBS at best-loss epoch failed: {e}")
                        best_ibs_at_best_epoch = float("inf")

            # -- Check HP result validity --
            hp_ibs = best_ibs_at_best_epoch
            if not np.isfinite(hp_ibs):
                reason = "NaN" if np.isnan(hp_ibs) else "Inf"
                logger.warning(f"Fold {fold} HP {ci+1}: val IBS at best epoch is {reason}, skipping")
                failed_combinations.append({
                    "fold": fold, "hp_index": ci, "params": params,
                    "reason": reason, "val_ibs": str(hp_ibs),
                })
                del model, optimizer, loss_fn, X_train_t, y_labels_t, X_val_t, x_cat_val, y_val_t
                del best_state_this_hp
                gc.collect()
                if _trim_at_hp: _malloc_trim()
                torch.cuda.empty_cache()
                continue

            metric_disp = f"val_loss={best_metric_this_hp:.4f}" if epoch_metric == "val_loss" else f"val_IBS={hp_ibs:.6f}"
            logger.info(f"  HP {ci+1} best: {metric_disp}, val_IBS={hp_ibs:.6f} at epoch {best_epoch_this_hp}")

            if hp_ibs < best_val_ibs:
                best_val_ibs = hp_ibs
                best_hp_idx = ci
                best_epoch_for_best_hp = best_epoch_this_hp
                best_state_for_best_hp = best_state_this_hp
                best_cfg_for_best_hp = deepcopy(dict(cfg))
                best_cfg_for_best_hp.pop("labtrans", None)
                best_labtrans_for_best_hp = labtrans

            del model, optimizer, loss_fn, X_train_t, y_labels_t, X_val_t, x_cat_val, y_val_t
            del best_state_this_hp
            gc.collect()
            if _trim_at_hp: _malloc_trim()
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
        logger.info(
            f"Fold {fold} WINNER: HP {best_hp_idx+1}, "
            f"val_IBS={best_val_ibs:.6f}, epoch={best_epoch_for_best_hp}, params={best_params}"
        )

        try:
            # Rebuild model and load best checkpoint
            t_refit_start = time.perf_counter()
            cfg_eval = EasyDict(deepcopy(best_cfg_for_best_hp))
            cfg_eval["labtrans"] = best_labtrans_for_best_hp
            cfg_eval["duration_index"] = best_labtrans_for_best_hp.cuts.astype(float)

            final_model = SurvTraceSingle(cfg_eval).to(device)
            if device.type == 'cuda':
                final_model.use_gpu = True
            final_model.load_state_dict(best_state_for_best_hp)
            final_model.eval()
            t_refit_end = time.perf_counter()
            fold_refit_seconds = t_refit_end - t_refit_start

            # Sanity check: recompute val IBS
            val_ibs_check = compute_val_ibs(
                final_model, df_val_X, Y_val, D_val,
                Y_train, D_train, time_grid,
                batch_size=int(best_params["batch_size"]),
            )
            logger.info(f"[CHECK] fold {fold}: HP-search best_val_IBS={best_val_ibs:.6f} vs loaded_val_IBS={val_ibs_check:.6f}")

            # Build trainfull for IPCW baselines
            y_trainfull = np.concatenate([y_train, y_val])
            Y_trainfull, D_trainfull = split_y(y_trainfull)
            time_grid_test = get_time_grid(df, y_trainfull, y_test)
            time_grid_test = subsample_time_grid(time_grid_test, ibs_grid_size, logger)

            # Baselines
            n_test = len(Y_test)
            t_test = time_grid_test.astype(float)

            random_ibs, km_ibs, km_mae_po = compute_baselines(
                Y_test, D_test, Y_trainfull, D_trainfull, t_test, n_test, logger, fold
            )

            batch_size_best = int(best_params["batch_size"])
            t_infer_start = time.perf_counter()
            surv_test_df = final_model.predict_surv_df(df_test_X, batch_size=batch_size_best)
            native_times_test = surv_test_df.index.values.astype(float)
            native_surv_test = surv_test_df.values.T  # (n_patients, n_times)
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

            logger.info(
                f"Fold {fold} TEST: IBS={test_ibs:.6f}, C-index={test_c:.6f}, "
                f"x_cal={x_cal_test:.6f}, p={p_value:.6g}, MAE_PO={mae_po_score:.6f}"
            )

            if save_plots:
                calib_path, km_path, surv_path = _save_plots(
                    fold, dataset_name, surv_test_df, surv_on_eval_grid,
                    time_grid_test, bin_stats, native_times_test, native_surv_test,
                    X_test, Y_test, D_test, Y_train, D_train, test_idx, seed,
                    metric_label="ibs",
                    plots_base=plots_base,
                )
                logger.info(f"Saved calibration plot: {calib_path}")
                logger.info(f"Saved KM curve plot: {km_path}")
                logger.info(f"Saved native vs interpolated survival curves: {surv_path}")
                plt.close('all')

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
            del best_state_for_best_hp
            X_train = X_val = X_test = None
            y_train = y_val = y_test = None
            Y_train = D_train = Y_val = D_val = Y_test = D_test = None
            y_trainfull = Y_trainfull = D_trainfull = None
            df_val_X = df_test_X = None
            surv_test_df = surv_on_eval_grid = None
            ev_test = ev_pycox = ev_random = ev_km = None
            random_surv = km_surv = km_func = None
            final_model = None
            best_cfg_for_best_hp = best_labtrans_for_best_hp = None
            gc.collect()
            if _trim_at_fold: _malloc_trim()
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - start_time

    """
    plot_dir = plots_base / 'cv_splits'
    plot_path = plot_cv_splits(
        fold_indices, len(X_all_np), n_folds,
        dataset_name, "SurvTrace", plot_dir, metric_label="ibs",
    )
    logger.info(f"Saved split visualization: {plot_path}")
    """

    (mean_test_ibs, std_test_ibs, mean_test_c, std_test_c,
     mean_random_ibs, mean_km_ibs, mean_mae_po, std_mae_po, mean_km_mae_po) = log_cv_summary(
        fold_results, logger, dataset_name, t_min, t_max, time_unit, n_folds, elapsed, n=len(df)
    )

    # Save JSON results
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = results_dir / f"kfold_survtrace_{dataset_name}_{device}_{ts}_ibs.json"

    details = {
        "dataset": dataset_name,
        "t_min": t_min,
        "t_max": t_max,
        "time_unit": time_unit,
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
        "hp_combinations": [{"hp_index": i, "params": c} for i, c in enumerate(combos)],
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
    }

    with open(details_path, "w") as f:
        json.dump(details, f, indent=2, default=str, separators=(',', ': '))
    logger.info(f"Saved results to: {details_path}")

    if save_plots:
        _save_training_curves(training_histories, dataset_name, seed, plots_base=plots_base)
        plt.close('all')

    return details_path, details

def _save_plots(
    fold, dataset_name, surv_test_df, surv_test_resampled,
    time_grid_test, bin_stats, native_times_test, native_surv_test,
    X_test, Y_test, D_test, Y_train, D_train, test_idx, seed,
    metric_label="ibs",
    plots_base=None,
):
    """Generate per-fold plots: calibration, KM, native vs interpolated survival curves."""
    if plots_base is None:
        plots_base = repo_root / 'Model_Training' / 'plots'
    calib_dir = plots_base / 'calibration_histogram' / 'survtrace'
    calib_path = save_calibration_histogram(bin_stats, dataset_name, "SurvTrace", fold, calib_dir, metric_label=metric_label)

    surv_np = surv_test_df.to_numpy()
    risk_scores = -surv_np.mean(axis=0)

    km_dir = plots_base / 'km_curves' / 'survtrace'
    km_path = plot_risk_stratified_km(
        risk_scores, Y_test, D_test, Y_train, D_train,
        dataset_name, "SurvTrace", fold, km_dir, metric_label=metric_label,
    )

    surv_dir = plots_base / 'individual_surv' / 'survtrace'
    surv_path = plot_native_vs_interpolated(
        native_times_test, native_surv_test, time_grid_test, surv_test_resampled,
        test_idx, dataset_name, "SurvTrace", fold, seed, surv_dir,
        metric_label=metric_label, n_curves=5, native_style="line",
    )
    return calib_path, km_path, surv_path


def _save_training_curves(training_histories, dataset_name, seed, plots_base=None):
    """Plot up to 9 random training curve histories."""
    if plots_base is None:
        plots_base = repo_root / 'Model_Training' / 'plots'
    if not training_histories:
        logger.warning("No training histories to plot")
        return
    n_to_plot = min(9, len(training_histories))
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(training_histories), size=n_to_plot, replace=False)

    n_rows = (n_to_plot + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i, hi in enumerate(selected):
        hist = training_histories[hi]
        ax = axes[i]
        ax2 = ax.twinx()

        epochs = hist.get("epochs", [])
        if not epochs:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        ax.plot(epochs, hist["train_loss"], 'b-', label='Train Loss', linewidth=1.5)
        ax.plot(epochs, hist["val_loss"], 'r-', label='Val Loss', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss', color='k')
        ax.legend(loc='upper left', fontsize=7)

        val_ibs = hist.get("val_ibs", [])
        ibs_epochs = [e for e, v in zip(epochs, val_ibs) if np.isfinite(v)]
        ibs_vals = [v for v in val_ibs if np.isfinite(v)]
        if ibs_vals:
            ax2.plot(ibs_epochs, ibs_vals, 'g--', label='Val IBS', linewidth=1.5, marker='o', markersize=3)
            ax2.set_ylabel('Val IBS', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.legend(loc='upper right', fontsize=7)

        ax.set_title(f"Fold {hist['fold']}, HP {hist['hp_index']+1}", fontsize=8)
        ax.grid(True, alpha=0.3)

    for i in range(n_to_plot, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    curves_dir = plots_base / 'training_curves' / 'survtrace'
    os.makedirs(curves_dir, exist_ok=True)
    ts_curves = datetime.now().strftime("%Y%m%d_%H%M%S")
    curves_path = f"{curves_dir}/training_curves_{dataset_name}_survtrace_ibs_{ts_curves}.png"
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training curves plot: {curves_path}")

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    datasets = ["ovarian", "breast", "micro.censure", "e1684", "acath", "prostateSurvival", "NSBCD", "nwtco", "nki70", "Z243", "burn", "veteran", "divorce", "UnempDur", "Unemployment", "support2"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--datasets', type=str, nargs='+', default=datasets)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--ibs_every', type=int, default=1,
                        help='Evaluate val IBS every N epochs (default: 1)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Stop training when epoch_metric has not improved for N eval rounds. '
                             'None (default) = train for all epochs.')
    parser.add_argument('--epoch_metric', type=str, default='val_ibs',
                        choices=['val_ibs', 'val_loss'],
                        help='Metric for epoch checkpointing and early stopping. '
                             'val_ibs = checkpoint best IBS epoch (default). '
                             'val_loss = checkpoint best loss epoch (like survhive).')
    parser.add_argument('--fold_time_budget_h', type=float, default=3.0,
                        help='Per-fold time budget in hours (default: 3.0)')
    parser.add_argument('--ibs_grid_size', type=int, default=None,
                        help='Subsample IBS time grid to this many evenly-spaced points (default: None = no subsampling)')
    parser.add_argument('--trim_memory', type=str, default='epoch',
                        choices=['epoch', 'hp', 'fold', 'none'],
                        help='When to call malloc_trim to return freed heap pages to OS. '
                             'none=never, epoch=every epoch, hp=after each HP combo, '
                             'fold=after each fold only.')
    parser.add_argument('--save_plots', action='store_true', default=False)
    args = parser.parse_args()

    device = args.device
    seed = 42

    logs_dir = EXPERIMENT_RESULTS / 'logs' / 'survtrace' / 'ibs'
    for dataset_name in args.datasets:
        os.makedirs(logs_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(
                    logs_dir / f"survtrace_kfold_ibs_{dataset_name}_{device}.log", mode="w",
                ),
                logging.StreamHandler(),
            ],
            force=True,
        )
        logger.info(f"Processing {dataset_name} on {device}")

        try:
            optimization_survtrace_kfold(
                dataset_name=dataset_name,
                seed=seed,
                device=device,
                n_folds=args.n_folds,
                ibs_every=args.ibs_every,
                patience=args.patience,
                epoch_metric=args.epoch_metric,
                fold_time_budget_h=args.fold_time_budget_h,
                ibs_grid_size=args.ibs_grid_size,
                trim_memory=args.trim_memory,
                save_plots=args.save_plots,
            )
        except Exception as e:
            logger.error(f"Failed on {dataset_name}: {e}", exc_info=True)
        finally:
            gc.collect()
            _malloc_trim()
            torch.cuda.empty_cache()
