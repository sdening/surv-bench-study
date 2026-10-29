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
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse
from sklearn.model_selection import train_test_split, StratifiedKFold

# survival specific imports
from coxkan import CoxKAN
from coxkan.utils import FastCoxLoss
try:
    from kan.LBFGS import LBFGS as KAN_LBFGS
except ImportError:
    KAN_LBFGS = None
from pycox.evaluation import EvalSurv

# local project imports
MAX_GRID_SAMPLES = 512

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
    get_native_surv_coxkan,
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

#helpers
def optimization_coxkan_kfold(
    dataset_name: str,
    seed: int,
    device: str = "cpu",
    n_folds: int = 5,
    TIME_COL: str = "time",
    EVENT_COL: str = "event",
    cindex_every: int = 10,
    patience: int = 10,
    fold_time_budget_h: float = 3.0,
    ibs_grid_size: int = None,
    save_plots: bool = False,
):
    logger.info(f"K-Fold CV CoxKAN C-index Script started (k={n_folds}) ")
    df = load_processed_dataset(
        dataset_name, method_name="coxkan", fix_split=False,
    )
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
    n_features = X_all_np.shape[1]

    logger.info(
        f"Dataset {dataset_name}: n={len(df)}, features={n_features}, "
        f"events={int(y_event_all.sum())}, "
        f"censor_rate={1.0 - float(y_event_all.mean()):.3f}"
    )

    # Load HP grid
    method_config_path = (
        repo_root / 'Model_Training' / 'method_configs' / 'coxkan.yaml'
    )
    with open(method_config_path, "r") as f:
        method_config = yaml.safe_load(f)

    grid_dict = method_config.get("hyperparameter_grid", {})
    keys = list(grid_dict.keys())
    combos = [
        dict(zip(keys, vals))
        for vals in itertools.product(*[grid_dict[k] for k in keys])
    ]
    logger.info(f"HP grid size: {len(combos)}")

    results_dir = EXPERIMENT_RESULTS / 'results' / 'coxkan' / 'cindex'
    plots_base = EXPERIMENT_RESULTS / 'plots' / 'coxkan' / 'cindex'

    kan_k = 3
    grid_update_num = 10
    stop_grid_update_step = 50

    start_time = time.perf_counter()
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []
    fold_indices = {}
    failed_combinations = []
    failed_folds = []
    training_histories = []

    FOLD_TIME_BUDGET_S = fold_time_budget_h * 3600
    budget_exceeded = False
    for fold, (trainfull_idx, test_idx) in enumerate(
        kfold.split(X_all_np, y_event_all), start=1
    ):
        logger.info(f"\n{'='*60}\nFold {fold}/{n_folds} | CoxKAN | C-index | {dataset_name}\n{'='*60}")

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

        logger.info(
            f"Train: {len(train_idx)} ({len(train_idx)/len(X_all_np)*100:.1f}%), "
            f"Val: {len(X_val)} ({len(X_val)/len(X_all_np)*100:.1f}%), "
            f"Test: {len(X_test)} ({len(X_test)/len(X_all_np)*100:.1f}%)"
        )

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_labels_t = torch.tensor(
            np.column_stack([Y_train, D_train]), dtype=torch.float32,
        )
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_labels_t = torch.tensor(
            np.column_stack([Y_val, D_val]), dtype=torch.float32,
        )

        n_train_samples = len(X_train_t)

        time_grid = get_time_grid(df, y_train, y_val)
        time_grid = subsample_time_grid(time_grid, ibs_grid_size, logger)
        if time_grid is None or len(time_grid) == 0:
            logger.warning(f"Fold {fold}: empty time grid, skipping")
            continue

        # HP search
        t_hp_start = time.perf_counter()
        best_hp_idx = None
        best_val_cindex = float("-inf")
        best_step_for_best_hp = None
        best_state_for_best_hp = None

        for ci, params in enumerate(combos):
            logger.info(f" Fold {fold} HP {ci+1}/{len(combos)}: {params}")

            lr = float(params.get("lr", 0.01))
            steps = int(params.get("steps", 100))
            base_fun = params.get("base_fun", "silu")
            opt_name = params.get("opt", "Adam")
            num_hidden = int(params.get("num_hidden", 0))
            hidden_dim = int(params.get("hidden_dim", 3))
            grid_size = int(params.get("grid", 3))
            batch_size = int(params.get("batch_size", 256))
            grid_update_freq = max(1, stop_grid_update_step // grid_update_num)
            eff_batch = min(batch_size, n_train_samples)

            if num_hidden > 0:
                width = [n_features] + [hidden_dim] * num_hidden + [1]
            else:
                width = [n_features, 1]

            torch.manual_seed(seed + fold)
            np.random.seed(seed + fold)

            model = CoxKAN(
                width=width, grid=grid_size, k=kan_k,
                seed=seed + fold, base_fun=base_fun,
            )

            if opt_name == "LBFGS" and KAN_LBFGS is not None:
                optimizer = KAN_LBFGS(
                    model.parameters(), lr=lr, history_size=10,
                    line_search_fn="strong_wolfe",
                    tolerance_grad=1e-32, tolerance_change=1e-32,
                    tolerance_ys=1e-32,
                )
            elif opt_name == "LBFGS":
                optimizer = torch.optim.LBFGS(
                    model.parameters(), lr=lr, history_size=10,
                    line_search_fn="strong_wolfe",
                )
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            best_cindex_this_hp = float("-inf")
            best_step_this_hp = None
            best_state_this_hp = None
            no_improve_count = 0

            history = {
                "fold": fold, "hp_index": ci, "params": params,
                "dataset": dataset_name,
                "steps": [], "train_loss": [], "val_loss": [],
                "val_cindex": [],
            }

            for step in range(steps):
                torch.nn.Module.train(model)

                if (step % grid_update_freq == 0
                        and step < stop_grid_update_step):
                    try:
                        if n_train_samples > MAX_GRID_SAMPLES:
                            _gidx = np.random.choice(
                                n_train_samples, MAX_GRID_SAMPLES, replace=False,
                            )
                            model.update_grid_from_samples(X_train_t[_gidx])
                        else:
                            model.update_grid_from_samples(X_train_t)
                    except Exception as e:
                        if step == 0:
                            logger.warning(f"Grid update failed step {step}: {e}")

                if opt_name == "LBFGS":
                    _loss_holder = [0.0]
                    def closure(_lh=_loss_holder):
                        optimizer.zero_grad()
                        pred = model(X_train_t)
                        loss = FastCoxLoss(pred, y_labels_t)
                        loss.backward()
                        _lh[0] = loss.item()
                        return loss
                    optimizer.step(closure)
                    step_loss = _loss_holder[0]
                elif eff_batch < n_train_samples:
                    perm = torch.randperm(n_train_samples)
                    batch_losses = []
                    for b0 in range(0, n_train_samples, eff_batch):
                        b_idx = perm[b0:b0 + eff_batch]
                        optimizer.zero_grad()
                        pred = model(X_train_t[b_idx])
                        loss = FastCoxLoss(pred, y_labels_t[b_idx])
                        if torch.isfinite(loss):
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=5.0,
                            )
                            optimizer.step()
                            batch_losses.append(loss.item())
                    step_loss = (
                        float(np.mean(batch_losses))
                        if batch_losses else float("nan")
                    )
                else:
                    optimizer.zero_grad()
                    pred = model(X_train_t)
                    loss = FastCoxLoss(pred, y_labels_t)
                    if torch.isfinite(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()
                        step_loss = loss.item()
                    else:
                        step_loss = float("nan")

                history["steps"].append(step + 1)
                history["train_loss"].append(step_loss)

                torch.nn.Module.train(model, False)
                with torch.no_grad():
                    val_pred = model(X_val_t)
                    val_loss = FastCoxLoss(val_pred, y_val_labels_t)
                    history["val_loss"].append(
                        val_loss.item() if torch.isfinite(val_loss) else float("nan")
                    )

                # Val C-index checkpoint
                if (step + 1) % cindex_every == 0:
                    torch.nn.Module.train(model, False)
                    with torch.no_grad():
                        try:
                            native_t_v, native_s_v = get_native_surv_coxkan(
                                model, X_train, Y_train, D_train, X_val,
                            )
                            surv_val_interp = interpolate_survival_to_grid(
                                native_t_v, native_s_v, time_grid,
                            )
                            surv_val_interp = np.clip(surv_val_interp, 1e-6, 1.0 - 1e-6)
                            surv_val_eval_df = pd.DataFrame(surv_val_interp.T, index=time_grid)
                            ev_val = EvalSurv(surv_val_eval_df, Y_val.astype(float), D_val.astype(float), steps='post')
                            cindex = float(ev_val.concordance_td(method='adj_antolini'))

                            if not np.isfinite(cindex) or cindex == 0:
                                history["val_cindex"].append(np.nan)
                                no_improve_count += 1
                            else:
                                logger.info(
                                    f"Fold {fold} HP {ci+1}/{len(combos)} "
                                    f"Step {step+1}: val_C-index={cindex:.6f}"
                                )
                                history["val_cindex"].append(cindex)
                                if cindex > best_cindex_this_hp:
                                    best_cindex_this_hp = cindex
                                    best_step_this_hp = step + 1
                                    best_state_this_hp = deepcopy(model.state_dict())
                                    no_improve_count = 0
                                else:
                                    no_improve_count += 1
                        except Exception as e:
                            logger.warning(f"Fold {fold} HP {ci+1} Step {step+1} eval failed: {e}")
                            history["val_cindex"].append(np.nan)
                            no_improve_count += 1
                else:
                    history["val_cindex"].append(np.nan)

                if patience is not None and no_improve_count >= patience:
                    logger.info(
                        f"Fold {fold} HP {ci+1}: early stopping at "
                        f"step {step+1} (no improvement for {patience} evals)"
                    )
                    break

            training_histories.append(history)

            if not np.isfinite(best_cindex_this_hp):
                failed_combinations.append({
                    "fold": fold, "hp_index": ci, "params": params,
                    "reason": "no finite val_C-index",
                    "val_cindex": str(best_cindex_this_hp),
                })
            elif best_cindex_this_hp > best_val_cindex:
                best_val_cindex = best_cindex_this_hp
                best_hp_idx = ci
                best_step_for_best_hp = best_step_this_hp
                best_state_for_best_hp = best_state_this_hp

            if 'model' in locals(): del model
            if 'optimizer' in locals(): del optimizer
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
        logger.info(
            f"Fold {fold} best HP: {best_params}, "
            f"val_C-index={best_val_cindex:.6f}, step={best_step_for_best_hp}"
        )

        # Rebuild best model and load checkpoint
        t_refit_start = time.perf_counter()
        best_num_hidden = int(best_params.get("num_hidden", 0))
        best_hidden_dim = int(best_params.get("hidden_dim", 3))
        best_grid_size = int(best_params.get("grid", 3))
        if best_num_hidden > 0:
            best_width = [n_features] + [best_hidden_dim] * best_num_hidden + [1]
        else:
            best_width = [n_features, 1]
        best_model = CoxKAN(
            width=best_width, grid=best_grid_size, k=kan_k,
            seed=seed + fold,
            base_fun=best_params.get("base_fun", "silu"),
        )
        best_model.load_state_dict(best_state_for_best_hp)
        torch.nn.Module.train(best_model, False)
        t_refit_end = time.perf_counter()
        fold_refit_seconds = t_refit_end - t_refit_start

        X_trainfull = np.concatenate([X_train, X_val])
        y_trainfull = np.concatenate([y_train, y_val])
        Y_trainfull, D_trainfull = split_y(y_trainfull)

        time_grid_test = get_time_grid(df, y_trainfull, y_test)
        time_grid_test = subsample_time_grid(time_grid_test, ibs_grid_size, logger)

        try:
            # Baselines
            n_test = len(Y_test)
            t_test = time_grid_test.astype(float)

            random_ibs, km_ibs, km_mae_po = compute_baselines(
                Y_test, D_test, Y_trainfull, D_trainfull, t_test, n_test, logger, fold
            )

            #  Test evaluation: native -> interpolate -> metrics
            t_infer_start = time.perf_counter()
            native_times_test, native_surv_test = get_native_surv_coxkan(
                best_model, X_trainfull, Y_trainfull, D_trainfull, X_test,
            )
            surv_on_eval_grid = interpolate_survival_to_grid(
                native_times_test, native_surv_test, time_grid_test,
            )
            surv_on_eval_grid = np.clip(surv_on_eval_grid, 1e-6, 1.0 - 1e-6)
            t_infer_end = time.perf_counter()
            fold_inference_seconds = t_infer_end - t_infer_start
            logger.info(
                f"Fold {fold} TIMING: HP search={fold_hp_search_seconds:.2f}s "
                f"({fold_hp_search_seconds/len(combos):.3f}s/combo), "
                f"model_load={fold_refit_seconds:.2f}s, "
                f"inference={fold_inference_seconds:.4f}s"
            )

            test_ibs, test_c, x_cal_test, p_value, bin_stats, mae_po_score = evaluate_on_test(
                surv_on_eval_grid, time_grid_test, Y_test, D_test, Y_trainfull, D_trainfull
            )

            logger.info(f"Fold {fold} TEST: IBS={test_ibs:.6f}, C-index={test_c:.6f}, x_cal={x_cal_test:.6f}, p={p_value:.6g}, MAE_PO={mae_po_score:.6f}")

            if save_plots:
                calib_path = save_calibration_histogram(bin_stats, dataset_name, "CoxKAN", fold, save_dir=plots_base / 'calibration_histogram' / 'coxkan', metric_label="cindex")
                logger.info(f"Saved calibration plot: {calib_path}")

                with torch.no_grad():
                    risk_scores = best_model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()
                km_path = plot_risk_stratified_km(risk_scores, Y_test, D_test, Y_train, D_train, dataset_name, "CoxKAN", fold, save_dir=plots_base / 'km_curves' / 'coxkan', metric_label="cindex")
                logger.info(f"Saved KM curve plot: {km_path}")

                try:
                    surv_path = plot_native_vs_interpolated(
                        native_times_test, native_surv_test, time_grid_test, surv_on_eval_grid,
                        test_idx, dataset_name, "CoxKAN", fold, seed,
                        save_dir=plots_base / 'individual_surv' / 'coxkan', metric_label="cindex", native_style="step",
                    )
                    logger.info(f"Saved individual survival curves: {surv_path}")
                except Exception as e:
                    logger.error(f"Failed to plot individual survival curves: {e}")

            fold_results.append({
                "fold": fold,
                "best_hp_index": best_hp_idx,
                "best_params": best_params,
                "best_step": best_step_for_best_hp,
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
                "fold_hp_search_seconds": fold_hp_search_seconds,
                "fold_hp_time_per_combo": fold_hp_search_seconds / len(combos),
                "fold_refit_seconds": fold_refit_seconds,
                "fold_inference_seconds": fold_inference_seconds,
            })
        except Exception as e:
            logger.error(f"Fold {fold} test phase failed: {e}", exc_info=True)
            failed_folds.append({"fold": fold, "reason": str(e)})
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - start_time

    # CV splits plot
    """
    plot_path = plot_cv_splits(fold_indices, len(X_all_np), n_folds, dataset_name, "CoxKAN", save_dir=plots_base / 'cv_splits', metric_label="cindex")
    logger.info(f"Saved split visualization: {plot_path}")
    """

    (mean_test_ibs, std_test_ibs, mean_test_c, std_test_c,
     mean_random_ibs, mean_km_ibs, mean_mae_po, std_mae_po, mean_km_mae_po) = log_cv_summary(
        fold_results, logger, dataset_name, t_min, t_max, time_unit, n_folds, elapsed, n=len(df)
    )

    # Save results JSON
    os.makedirs(results_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = results_dir / f"kfold_coxkan_{dataset_name}_{ts}_cindex.json"

    details = {
        "dataset": dataset_name,
        "seed": seed,
        "n_folds": n_folds,
        "optimization_metric": "C-index",
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
        "kan_defaults": {
            "k": kan_k,
            "grid_update_num": grid_update_num,
            "stop_grid_update_step": stop_grid_update_step,
        },
        "memory_scaling": {
            "max_grid_samples": MAX_GRID_SAMPLES,
        },
        "hp_combinations": [
            {"hp_index": i, "params": combo}
            for i, combo in enumerate(combos)
        ],
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

                steps_arr = hist["steps"]
                ax.plot(steps_arr, hist["train_loss"], 'b-', label='Train Loss', linewidth=1.5)
                ax.plot(steps_arr, hist["val_loss"], 'r-', label='Val Loss', linewidth=1.5)
                ax.set_xlabel('Step')
                ax.set_ylabel('Loss', color='k')
                ax.tick_params(axis='y', labelcolor='k')

                cindex_steps = [s for s, v in zip(steps_arr, hist["val_cindex"]) if not np.isnan(v)]
                cindex_values = [v for v in hist["val_cindex"] if not np.isnan(v)]
                if cindex_values:
                    ax2.plot(cindex_steps, cindex_values, 'g--', label='Val C-index', linewidth=1.5, marker='o', markersize=3)
                    ax2.set_ylabel('Val C-index', color='g')
                    ax2.tick_params(axis='y', labelcolor='g')

                title = f"Fold {hist['fold']}, {hist['dataset']}\nHP {hist['hp_index']+1}: {hist['params']}"
                ax.set_title(title, fontsize=8)
                ax.legend(loc='upper left', fontsize=7)
                if cindex_values:
                    ax2.legend(loc='upper right', fontsize=7)
                ax.grid(True, alpha=0.3)

            for idx in range(len(selected_indices), 9):
                axes[idx].axis('off')

            plt.tight_layout()
            curves_dir = plots_base / 'training_curves' / 'coxkan'
            os.makedirs(curves_dir, exist_ok=True)
            ts_curves = datetime.now().strftime("%Y%m%d_%H%M%S")
            curves_path = curves_dir / f"training_curves_{dataset_name}_coxkan_cindex_{ts_curves}.png"
            plt.savefig(curves_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved training curves: {curves_path}")

    return details_path, details


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="CoxKAN K-fold CV with C-index optimization ",
    )
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=["ovarian", "breast", "micro.censure", "e1684", "acath", "prostateSurvival", "NSBCD", "nwtco", "nki70", "Z243", "burn", "veteran", "divorce", "UnempDur", "Unemployment", "support2"])
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--cindex_every', type=int, default=1,
                        help='Evaluate val C-index every N training steps')
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

    logs_dir = EXPERIMENT_RESULTS / 'logs' / 'coxkan' / 'cindex'
    for dataset_name in args.datasets:
        os.makedirs(logs_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(
                    logs_dir / f"coxkan_kfold_cindex_{dataset_name}.log", mode="w",
                ),
                logging.StreamHandler(),
            ],
            force=True,
        )

        logger.info(f"\nProcessing {dataset_name}")
        logger.info(f"Using device: {device}")

        try:
            optimization_coxkan_kfold(
                dataset_name=dataset_name,
                seed=seed,
                device=device,
                n_folds=args.n_folds,
                cindex_every=args.cindex_every,
                patience=args.patience,
                fold_time_budget_h=args.fold_time_budget_h,
                ibs_grid_size=args.ibs_grid_size,
                save_plots=args.save_plots,
            )
        except Exception as e:
            logger.error(f"Failed: {e}", exc_info=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
