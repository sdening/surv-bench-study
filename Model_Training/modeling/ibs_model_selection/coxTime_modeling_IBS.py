import gc
import ctypes
import json
import os
import sys
import sysconfig
import matplotlib.pyplot as plt
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
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch

def _malloc_trim():
    """Force glibc to return freed heap pages to the OS (Linux only)."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
from sklearn.model_selection import StratifiedKFold

# survival specific imports
from pycox.models import CoxTime
from pycox.models.data import CoxTimeDataset
from SurvivalEVAL.Evaluator import SurvivalEvaluator

if sys.platform == "darwin" and "_OMP_REEXEC" not in os.environ: # Fix for OMP + PyTorch on MacOS (?)
    _sp = sysconfig.get_paths()["purelib"]
    _torch_lib = os.path.join(_sp, "torch", "lib")
    if os.path.isdir(_torch_lib):
        os.environ["DYLD_LIBRARY_PATH"] = _torch_lib
        os.environ["_OMP_REEXEC"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
warnings.filterwarnings('ignore', category=RuntimeWarning, module='SurvivalEVAL')

if not hasattr(pd.Series, "is_monotonic"):
    pd.Series.is_monotonic = property(lambda self: self.is_monotonic_increasing)

if not hasattr(pd.Series, "iteritems"):
    pd.Series.iteritems = pd.Series.items


class CoxTimeBaseNeuralNetWrapper(nn.Module):
    def __init__(self, base_neural_net):
        super(CoxTimeBaseNeuralNetWrapper, self).__init__()
        self.net = base_neural_net

    def forward(self, inputs_excluding_times, times):
        if times.dim() == 1:
            times = times.unsqueeze(1)
        inputs = torch.cat([inputs_excluding_times, times], dim=1)
        return self.net(inputs)


def scaled_to_orig_continuous(labtrans, scaled_times: np.ndarray) -> np.ndarray:
    z = np.asarray(scaled_times, dtype=np.float32).reshape(-1, 1)
    t = labtrans.duration_scaler.inverse_transform(z).reshape(-1)
    if labtrans.log_duration:
        t = np.expm1(t)
    return t


def optimization_coxtime_kfold(
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
    K-fold CV for CoxTime (IBS optimised, standardised interpolation).
    - Split data into k folds using StratifiedKFold
    - Per fold: 80% train_full, 20% test
    - Split train_full into 80% train, 20% val
    - Final split per fold: 64% train, 16% val, 20% test
    - Find best HP combo on val (lowest IBS + best epoch)
    - Evaluate that model on test
    - Average performance over k test sets
    """

    if device not in ['cpu', 'cuda'] and not device.startswith('cuda:'):
        raise ValueError(f"Invalid device: {device}")
    device = torch.device(device)

    logger.info(f"K-Fold CV CoxTime IBS Script started (k={n_folds}) ")
    df = load_processed_dataset(dataset_name, method_name="coxtime", fix_split=False)
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

    if device.type == 'cuda':
        X_all_gpu = torch.tensor(X_all_np, dtype=torch.float32, device=device)
        y_time_gpu = torch.tensor(y_all["time"].copy(), dtype=torch.float32, device=device)
        y_event_gpu = torch.tensor(y_all["event"].copy().astype(int), dtype=torch.int64, device=device)
        logger.info(f"[GPU] Pre-loaded dataset to GPU")
    else:
        X_all_gpu = y_time_gpu = y_event_gpu = None

    logger.info(f"Dataset {dataset_name}: n={len(df)}, events={int(y_event_all.sum())}, censor_rate={1.0 - float(y_event_all.mean()):.3f}")

    # Load HP grid
    grid_dict, combos = load_hp_grid(repo_root, 'coxtime', logger)

    results_dir = EXPERIMENT_RESULTS / 'results' / 'coxtime' / 'ibs'
    plots_base = EXPERIMENT_RESULTS / 'plots' / 'coxtime' / 'ibs'

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
        logger.info(f"\n{'='*60}\nFold {fold}/{n_folds} | CoxTime | IBS | {dataset_name}\n{'='*60}")

        trainfull_idx = np.asarray(trainfull_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)

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

        X_train, y_train = X_all_np[train_idx], y_all[train_idx]
        X_val, y_val = X_all_np[val_idx], y_all[val_idx]
        X_test, y_test = X_all_np[test_idx], y_all[test_idx]

        # Containment fix
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

        time_grid = get_time_grid(df, y_train, y_val)
        time_grid = subsample_time_grid(time_grid, ibs_grid_size, logger)
        if time_grid is None or len(time_grid) == 0:
            logger.warning(f"Fold {fold}: empty time grid, skipping")
            continue

        #HP search
        best_hp_idx = None
        best_val_ibs = float("inf")
        best_epoch_for_best_hp = None
        best_state_for_best_hp = None
        best_labtrans_for_best_hp = None

        t_hp_start = time.perf_counter()
        for ci, params in enumerate(combos):
            logger.info(f" Fold {fold} HP {ci+1}/{len(combos)}: {params}")

            torch.manual_seed(seed + fold)
            np.random.seed(seed + fold)
            hidden = int(params.get("hidden", 8))
            lr = float(params.get("learning_rate", params.get("lr", 1e-3)))
            epochs = int(params.get("epochs", 100))
            batch_size = int(params.get("batch_size", 128))
            dropout = float(params.get("dropout", 0.0))

            # CoxTime label transform (fit on train only)
            label_transform = CoxTime.label_transform()
            Y_train_np, D_train_np = label_transform.fit_transform(Y_train.astype(np.float32), D_train.astype(np.int32))
            Y_val_np, D_val_np = label_transform.transform(Y_val.astype(np.float32), D_val.astype(np.int32))
            Y_test_np, D_test_np = label_transform.transform(Y_test.astype(np.float32), D_test.astype(np.int32))

            train_sort_idx = np.argsort(Y_train_np)

            # CoxTimeDataset (must be sorted by time) -- case/control structure
            train_data = CoxTimeDataset(
                X_train[train_sort_idx].astype('float32', copy=False),
                Y_train_np[train_sort_idx].astype('float32', copy=False),
                D_train_np[train_sort_idx].astype('int32', copy=False),
            )

            val_sort_indices = np.argsort(Y_val_np)
            val_data = CoxTimeDataset(
                X_val[val_sort_indices].astype('float32', copy=False),
                Y_val_np[val_sort_indices].astype('float32', copy=False),
                D_val_np[val_sort_indices].astype('int32', copy=False),
            )

            #num_val_set_repeats = 5
            #X_val_repeated_np = np.tile(X_val, (num_val_set_repeats, 1))
            #Y_val_repeated_np = np.tile(Y_val_np, num_val_set_repeats)
            #D_val_repeated_np = np.tile(D_val_np, num_val_set_repeats)

            #val_sort_indices = np.argsort(Y_val_repeated_np)
            #val_data = CoxTimeDataset(
            #    X_val_repeated_np[val_sort_indices].astype('float32'),
            #    Y_val_repeated_np[val_sort_indices].astype('float32'),
            #    D_val_repeated_np[val_sort_indices].astype('int32'),
            #)

            # Build model (time is an additional input feature)
            layers = [nn.Linear(X_train.shape[1] + 1, hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(hidden, 1))
            base_neural_net = nn.Sequential(*layers).to(device)

            base_neural_net_wrapper = CoxTimeBaseNeuralNetWrapper(base_neural_net).to(device)
            #coxtime_model = CoxTime(base_neural_net_wrapper, device=device, labtrans=label_transform)
            coxtime_model = CoxTime(base_neural_net_wrapper, device=device, labtrans=None)
            coxtime_loss = coxtime_model.loss
            optimizer = torch.optim.Adam(base_neural_net_wrapper.parameters(), lr=lr)

            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

            best_ibs_this_hp = float("inf")
            best_epoch_this_hp = None
            best_state_this_hp = None
            best_labtrans_this_hp = None
            no_improve_count = 0

            history = {
                "fold": fold, "hp_index": ci, "params": params,
                "dataset": dataset_name,
                "epochs": [], "train_loss": [], "val_loss": [], "val_ibs": [],
            }

            for epoch in range(1, epochs + 1):
                # Training (case/control) (see: https://github.com/georgehc/survival-intro/blob/main/S3.4_Cox-Time.ipynb)
                base_neural_net_wrapper.train()
                epoch_train_losses = []
                for case, controls in train_loader:
                    case_X_batch = case[0].squeeze(1).to(device)
                    case_Y_batch = case[1].squeeze(1).to(device)
                    control_X_batch = controls[0][0].squeeze(1).to(device)
                    control_Y_batch = controls[0][1].squeeze(1).to(device)

                    case_neural_net_output = base_neural_net_wrapper(case_X_batch, case_Y_batch)
                    control_neural_net_output = base_neural_net_wrapper(control_X_batch, control_Y_batch)
                    loss_batch = coxtime_loss(case_neural_net_output, control_neural_net_output)

                    if torch.isfinite(loss_batch):
                        epoch_train_losses.append(loss_batch.item())
                        optimizer.zero_grad(set_to_none=True)
                        loss_batch.backward()
                        torch.nn.utils.clip_grad_norm_(base_neural_net_wrapper.parameters(), max_norm=5.0)
                        optimizer.step()

                # validation loss (case/control)
                base_neural_net_wrapper.eval()
                epoch_val_losses = []
                with torch.inference_mode():
                    for case, controls in val_loader:
                        case_X_batch = case[0].squeeze(1).to(device)
                        case_Y_batch = case[1].squeeze(1).to(device)
                        case_neural_net_output = base_neural_net_wrapper(case_X_batch, case_Y_batch)
                        control_X_batch = controls[0][0].squeeze(1).to(device)
                        control_Y_batch = controls[0][1].squeeze(1).to(device)
                        control_neural_net_output = base_neural_net_wrapper(control_X_batch, control_Y_batch)
                        loss_batch = coxtime_loss(case_neural_net_output, control_neural_net_output)
                        if torch.isfinite(loss_batch):
                            epoch_val_losses.append(loss_batch.item())

                if epoch_train_losses:
                    history["epochs"].append(epoch)
                    history["train_loss"].append(np.mean(epoch_train_losses))
                    history["val_loss"].append(np.mean(epoch_val_losses) if epoch_val_losses else np.nan)

                #ibs on val
                if epoch % ibs_every == 0:
                    base_neural_net_wrapper.eval()
                    with torch.inference_mode():
                        try:
                            n_train_sorted = len(train_sort_idx)
                            coxtime_model.compute_baseline_hazards(
                                input=X_train.astype("float32")[train_sort_idx],
                                target=(Y_train_np[train_sort_idx].astype("float32"),
                                        D_train_np[train_sort_idx].astype("int32")),
                                sample=min(500, n_train_sorted),
                            )
                            surv_val_df = coxtime_model.predict_surv_df(
                                X_val.astype("float32"), batch_size=batch_size,
                            )
                            surv_val_df.index = scaled_to_orig_continuous(
                                label_transform, surv_val_df.index.values,
                            )

                            native_times_v = surv_val_df.index.values
                            native_surv_v = surv_val_df.values.T
                            surv_val_interp = interpolate_survival_to_grid(
                                native_times_v, native_surv_v, time_grid,
                            )
                            surv_val_interp = np.clip(surv_val_interp, 1e-6, 1.0 - 1e-6)

                            ev_val = SurvivalEvaluator(
                                pred_survs=surv_val_interp,
                                time_coordinates=time_grid.astype(float),
                                event_times=Y_val, event_indicators=D_val,
                                train_event_times=Y_train, train_event_indicators=D_train,
                            )
                            ibs = float(ev_val.integrated_brier_score(
                                num_points=None, target_times=time_grid.astype(float),
                                IPCW_weighted=True, integration_method="trapz", draw_figure=False,
                            ))

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
                                    best_state_this_hp = deepcopy(base_neural_net_wrapper.state_dict())
                                    best_labtrans_this_hp = deepcopy(label_transform)
                                    no_improve_count = 0
                                else:
                                    no_improve_count += 1

                        except Exception as e:
                            logger.warning(f"Fold {fold} HP {ci+1} Epoch {epoch} failed: {e}")
                            history["val_ibs"].append(np.nan)
                            no_improve_count += 1
                        finally:
                            surv_val_df = native_times_v = native_surv_v = None
                            surv_val_interp = ev_val = None
                else:
                    history["val_ibs"].append(np.nan)

                if patience is not None and no_improve_count >= patience:
                    logger.info(
                        f"Fold {fold} HP {ci+1}: early stopping at "
                        f"epoch {epoch} (no improvement for {patience} evals)"
                    )
                    break

            training_histories.append(history)

            if not np.isfinite(best_ibs_this_hp):
                reason = "NaN" if np.isnan(best_ibs_this_hp) else "Inf"
                logger.warning(f"Fold {fold} HP {ci+1}: best val_IBS is {reason}, skipping")
                failed_combinations.append({
                    "fold": fold, "hp_index": ci, "params": params,
                    "reason": f"best val_IBS is {reason}",
                    "val_ibs": str(best_ibs_this_hp),
                })
            elif best_ibs_this_hp < best_val_ibs:
                best_val_ibs = best_ibs_this_hp
                best_hp_idx = ci
                best_epoch_for_best_hp = best_epoch_this_hp
                best_state_for_best_hp = best_state_this_hp
                best_labtrans_for_best_hp = best_labtrans_this_hp

            if 'base_neural_net' in locals(): del base_neural_net
            if 'base_neural_net_wrapper' in locals(): del base_neural_net_wrapper
            if 'coxtime_model' in locals(): del coxtime_model
            if 'optimizer' in locals(): del optimizer
            if 'train_loader' in locals(): del train_loader
            if 'val_loader' in locals(): del val_loader
            if 'train_data' in locals(): del train_data
            if 'val_data' in locals(): del val_data

            if 'best_state_this_hp' in locals(): del best_state_this_hp
            if 'best_labtrans_this_hp' in locals(): del best_labtrans_this_hp
            gc.collect()
            _malloc_trim()
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
            # Rebuild best model, load weights, compute baseline hazards on train+val
            t_refit_start = time.perf_counter()

            batch_size_best = int(best_params.get("batch_size", 128))
            hidden = int(best_params.get("hidden", 8))
            dropout = float(best_params.get("dropout", 0.0))

            layers = [nn.Linear(X_train.shape[1] + 1, hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(hidden, 1))
            base_neural_net = nn.Sequential(*layers).to(device)

            net = CoxTimeBaseNeuralNetWrapper(base_neural_net).to(device)
            net.load_state_dict(best_state_for_best_hp)
            net.eval()

            final_model = CoxTime(net, device=device, labtrans=None)

            y_trainfull = np.concatenate([y_train, y_val])
            X_trainfull = np.concatenate([X_train, X_val], axis=0)
            Y_trainfull, D_trainfull = split_y(y_trainfull)

            Y_trainfull_ct, D_trainfull_ct = best_labtrans_for_best_hp.transform(
                Y_trainfull.astype(np.float32), D_trainfull.astype(np.int32),
            )
            idx_full = np.argsort(Y_trainfull_ct)

            n_trainfull = len(idx_full)
            final_model.compute_baseline_hazards(
                input=X_trainfull[idx_full].astype("float32", copy=False),
                target=(Y_trainfull_ct[idx_full].astype("float32", copy=False),
                        D_trainfull_ct[idx_full].astype("int32", copy=False)),
                sample=min(500, n_trainfull),
            )
            t_refit_end = time.perf_counter()
            fold_refit_seconds = t_refit_end - t_refit_start

            time_grid_test = get_time_grid(df, y_trainfull, y_test)
            time_grid_test = subsample_time_grid(time_grid_test, ibs_grid_size, logger)

            # Baselines
            n_test = len(Y_test)
            t_test = time_grid_test.astype(float)

            random_ibs, km_ibs, km_mae_po = compute_baselines(
                Y_test, D_test, Y_trainfull, D_trainfull, t_test, n_test, logger, fold
            )

            # Test evaluation: native -> interpolate -> metrics
            t_infer_start = time.perf_counter()
            surv_test_df = final_model.predict_surv_df(
                X_test.astype("float32"), batch_size=batch_size_best,
            )
            surv_test_df.index = scaled_to_orig_continuous(
                best_labtrans_for_best_hp, surv_test_df.index.values,
            )

            native_times_test = surv_test_df.index.values
            native_surv_test = surv_test_df.values.T
            surv_on_eval_grid = interpolate_survival_to_grid(
                native_times_test, native_surv_test, time_grid_test,
            )
            surv_on_eval_grid = np.clip(surv_on_eval_grid, 1e-6, 1.0 - 1e-6)
            t_infer_end = time.perf_counter()
            fold_inference_seconds = t_infer_end - t_infer_start
            logger.info(
                f"Fold {fold} TIMING: HP search={fold_hp_search_seconds:.2f}s "
                f"({fold_hp_search_seconds/len(combos):.3f}s/combo), "
                f"refit/model_load={fold_refit_seconds:.2f}s, "
                f"inference={fold_inference_seconds:.4f}s"
            )

            test_ibs, test_c, x_cal_test, p_value, bin_stats, mae_po_score = evaluate_on_test(
                surv_on_eval_grid, time_grid_test, Y_test, D_test, Y_trainfull, D_trainfull
            )

            if save_plots:
                calib_path = save_calibration_histogram(
                    bin_stats, dataset_name, "CoxTime", fold,
                    save_dir=plots_base / 'calibration_histogram' / 'coxtime', metric_label="ibs",
                )
                logger.info(f"Saved calibration plot: {calib_path}")

                t_star = float(np.median(Y_trainfull))
                t_star = float(np.clip(t_star, time_grid_test[0], time_grid_test[-1]))
                t_star_idx = np.searchsorted(time_grid_test, t_star, side="right") - 1
                t_star_idx = int(np.clip(t_star_idx, 0, len(time_grid_test) - 1))
                surv_at_t = surv_on_eval_grid[:, t_star_idx]
                risk_scores = -np.log(np.clip(surv_at_t, 1e-6, 1.0))

                km_path = plot_risk_stratified_km(
                    risk_scores, Y_test, D_test, Y_trainfull, D_trainfull,
                    dataset_name, "CoxTime", fold,
                    save_dir=plots_base / 'km_curves' / 'coxtime', metric_label="ibs",
                )
                logger.info(f"Saved KM curve plot: {km_path}")

                surv_path = plot_native_vs_interpolated(
                    native_times_test, native_surv_test, time_grid_test, surv_on_eval_grid,
                    test_idx, dataset_name, "CoxTime", fold, seed,
                    save_dir=plots_base / 'individual_surv' / 'coxtime', metric_label="ibs",
                    native_style="step",
                )
                logger.info(f"Saved native vs interpolated survival curves: {surv_path}")

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
            del best_state_for_best_hp
            X_train = X_val = X_test = None
            y_train = y_val = y_test = None
            Y_train = D_train = Y_val = D_val = Y_test = D_test = None
            X_trainfull = y_trainfull = Y_trainfull = D_trainfull = None
            X_train_t = Y_train_t = D_train_t = X_val_t = Y_val_t = D_val_t = None
            Y_trainfull_ct = D_trainfull_ct = None
            surv_test_df = surv_on_eval_grid = surv_eval_df = None
            ev_test = ev_pycox = ev_random = ev_km = None
            random_surv = km_surv = km_func = None
            final_model = net = base_neural_net = None
            best_labtrans_for_best_hp = None
            gc.collect()
            _malloc_trim()
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - start_time

    """
    plot_path = plot_cv_splits(
        fold_indices, len(X_all_np), n_folds,
        dataset_name, "CoxTime",
        save_dir=plots_base / 'cv_splits', metric_label="ibs",
    )
    logger.info(f"Saved split visualization: {plot_path}")
    """

    (mean_test_ibs, std_test_ibs, mean_test_c, std_test_c,
     mean_random_ibs, mean_km_ibs, mean_mae_po, std_mae_po, mean_km_mae_po) = log_cv_summary(
        fold_results, logger, dataset_name, t_min, t_max, time_unit, n_folds, elapsed, n=len(df)
    )

    # Save results
    os.makedirs(results_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = results_dir / f"kfold_coxtime_{dataset_name}_{ts}_ibs.json"

    details = {
        "dataset": dataset_name,
        "seed": seed,
        "n_folds": n_folds,
        "optimization_metric": "IBS",
        "t_min": t_min,
        "t_max": t_max,
        "time_unit": time_unit,
        "cindex_method": "pycox_adj_antolini",
        "interpolation": "linear (standardized)",
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

                hist_epochs = hist["epochs"]
                ax.plot(hist_epochs, hist["train_loss"], 'b-', label='Train Loss', linewidth=1.5)
                ax.plot(hist_epochs, hist["val_loss"], 'r-', label='Val Loss', linewidth=1.5)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss', color='k')
                ax.tick_params(axis='y', labelcolor='k')

                ibs_epochs = []
                ibs_values = []
                for j, e in enumerate(hist_epochs):
                    v = hist["val_ibs"][j]
                    if np.isfinite(v):
                        ibs_epochs.append(e)
                        ibs_values.append(v)

                if ibs_values:
                    ax2.plot(ibs_epochs, ibs_values, 'g--', label='Val IBS',
                             linewidth=1.5, marker='o', markersize=3)
                    ax2.set_ylabel('IBS', color='g')
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
            curves_dir = plots_base / 'training_curves' / 'coxtime'
            os.makedirs(curves_dir, exist_ok=True)
            ts_curves = datetime.now().strftime("%Y%m%d_%H%M%S")
            curves_path = curves_dir / f"training_curves_{dataset_name}_coxtime_ibs_{ts_curves}.png"
            plt.savefig(curves_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved training curves: {curves_path}")

    return details_path, details


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=["ovarian", "breast", "micro.censure", "e1684", "acath",
                                 "prostateSurvival", "NSBCD", "nwtco", "nki70", "Z243",
                                 "burn", "veteran", "divorce", "UnempDur", "Unemployment", "support2"],
                        )
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
    seed = 42

    for dataset_name in args.datasets:
        logger.info(f"\nProcessing {dataset_name}")

        logs_dir = EXPERIMENT_RESULTS / 'logs' / 'coxtime' / 'ibs'
        os.makedirs(logs_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(logs_dir / f"coxtime_kfold_ibs_{dataset_name}.log", mode="w"),
                logging.StreamHandler()
            ],
            force=True
        )

        try:
            optimization_coxtime_kfold(
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
            _malloc_trim()
            torch.cuda.empty_cache()
