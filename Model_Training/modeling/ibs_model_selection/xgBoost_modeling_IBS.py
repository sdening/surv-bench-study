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

from SurvivalEVAL.Evaluator import SurvivalEvaluator
from xgbse.converters import convert_data_to_xgb_format
import xgboost as xgb

def get_repo_root():
    return Path(__file__).resolve().parent.parent.parent.parent

repo_root = get_repo_root()
EXPERIMENT_RESULTS = repo_root / 'Model_Training' / 'modeling' / 'experiment_results'
sys.path.append(str(repo_root / 'Data_Preprocessing' / 'src'))
sys.path.append(str(repo_root / 'Model_Training' / 'src'))
from modeling_utils import (
    get_time_grid, split_y, load_processed_dataset,
    interpolate_survival_to_grid, apply_containment_fix, subsample_time_grid,
    compute_baselines, evaluate_on_test, log_cv_summary,
    get_native_surv_xgb_cox,
    get_native_surv_xgb_aft,
)
from evaluation_utils import surv_df_from_aft_mu
from plot_utils import (
    plot_native_vs_interpolated,
    plot_risk_stratified_km,
    save_calibration_histogram,
    plot_xgb_sweep_curves,
)

warnings.filterwarnings('ignore', category=UserWarning, module='SurvivalEVAL')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='SurvivalEVAL')

if not hasattr(pd.Series, "is_monotonic"):
    pd.Series.is_monotonic = property(lambda self: self.is_monotonic_increasing)


def evaluate_ibs_on_surv(native_times, native_surv, time_grid,
                         Y_eval, D_eval, Y_train, D_train):
    """Interpolate survival curves to grid and compute IBS."""
    surv_interp = interpolate_survival_to_grid(native_times, native_surv, time_grid)
    surv_interp = np.clip(surv_interp, 1e-6, 1.0 - 1e-6)
    ev = SurvivalEvaluator(
        pred_survs=surv_interp,
        time_coordinates=time_grid.astype(float),
        event_times=Y_eval, event_indicators=D_eval,
        train_event_times=Y_train, train_event_indicators=D_train,
    )
    ibs = float(ev.integrated_brier_score(
        num_points=None, target_times=time_grid.astype(float),
        IPCW_weighted=True, integration_method="trapz", draw_figure=False,
    ))
    return ibs, surv_interp



def eval_checkpoint_ibs(model, n_rounds, xgb_loss, params,
                        dtrain, dval, y_train, time_grid,
                        Y_val, D_val, Y_train_split, D_train_split):
    """Evaluate model[:n_rounds] on the validation set. Returns (ibs, ok)."""
    try:
        sliced = model[:n_rounds]
        if xgb_loss == "aft":
            mu = sliced.predict(dval, output_margin=True)
            dist = params.get("aft_loss_distribution", "normal")
            scale = params.get("aft_loss_distribution_scale", 1.0)
            ns = surv_df_from_aft_mu(mu, time_grid, dist=dist, scale=scale)
            ns = np.clip(ns, 1e-6, 1.0 - 1e-6)
            ev = SurvivalEvaluator(
                pred_survs=ns, time_coordinates=time_grid.astype(float),
                event_times=Y_val, event_indicators=D_val,
                train_event_times=Y_train_split, train_event_indicators=D_train_split,
            )
            ibs = float(ev.integrated_brier_score(
                num_points=None, target_times=time_grid.astype(float),
                IPCW_weighted=True, integration_method="trapz", draw_figure=False,
            ))
            del ev
        else:
            lp_val = sliced.predict(dval, output_margin=True)
            lp_tr = sliced.predict(dtrain, output_margin=True)
            nt, ns = get_native_surv_xgb_cox(y_train, lp_tr, lp_val)
            ibs, _ = evaluate_ibs_on_surv(
                nt, ns, time_grid, Y_val, D_val, Y_train_split, D_train_split,
            )
        del sliced, ns
        gc.collect()
        return (ibs, True) if np.isfinite(ibs) else (float("nan"), False)
    except Exception:
        return float("nan"), False


def select_best_checkpoint(
    combos, objective, xgb_loss,
    dtrain, dval, y_train, time_grid,
    Y_val, D_val, Y_train_split, D_train_split,
    max_n_rounds, round_step, seed, fold,
    fold_time_budget_s, logger, patience=None,
):
    """sweep across all HP combos 
    """
    best_hp_idx = None
    best_n_rounds = None
    best_val_ibs = float("inf")
    best_model = None
    sweep_log = []
    failed = []
    budget_exceeded = False

    t0 = time.perf_counter()

    for ci, params in enumerate(combos):
        logger.info(f" Fold {fold} HP {ci+1}/{len(combos)}: {params}")
        train_params = {**params, "objective": objective, "seed": seed + fold}
        model = xgb.train(train_params, dtrain, num_boost_round=max_n_rounds)

        combo_best_ibs = float("inf")
        no_improve_count = 0

        for r in range(round_step, max_n_rounds + 1, round_step):
            ibs, ok = eval_checkpoint_ibs(
                model, r, xgb_loss, params,
                dtrain, dval, y_train, time_grid,
                Y_val, D_val, Y_train_split, D_train_split,
            )
            if not ok:
                failed.append({"fold": fold, "hp_index": ci, "n_rounds": r,
                               "params": params, "reason": "eval failed"})
                continue
            sweep_log.append({"hp_idx": ci, "n_rounds": r,
                              "val_ibs": ibs, "params": params})

            if ibs < combo_best_ibs:
                combo_best_ibs = ibs
                no_improve_count = 0
            else:
                no_improve_count += 1

            if ibs < best_val_ibs:
                best_val_ibs = ibs
                best_hp_idx = ci
                best_n_rounds = r
                best_model = model

            if patience is not None and no_improve_count >= patience:
                logger.info(
                    f" Fold {fold} HP {ci+1}: patience exhausted at round {r} "
                    f"(no improvement for {patience} checkpoints)"
                )
                break

        if best_model is not model:
            del model

        logger.info(
            f" Fold {fold} HP {ci+1} done. "
            f"Best so far: HP {best_hp_idx}, round {best_n_rounds}, "
            f"val_IBS={best_val_ibs:.6f}"
        )

        if time.perf_counter() - t0 > fold_time_budget_s:
            logger.warning(
                f"Fold {fold}: TIME BUDGET EXCEEDED after {ci+1}/{len(combos)} combos."
            )
            budget_exceeded = True
            break

    gc.collect()
    hp_search_seconds = time.perf_counter() - t0
    return (best_model, best_hp_idx, best_n_rounds, best_val_ibs,
            sweep_log, failed, budget_exceeded, hp_search_seconds)


def optimization_xgboost_kfold(
    dataset_name: str,
    seed: int,
    n_folds: int = 5,
    TIME_COL: str = "time",
    EVENT_COL: str = "event",
    xgb_loss: str = "aft",
    time_unit: str = None,
    fold_time_budget_h: float = 3.0,
    max_n_rounds: int = 100,
    round_step: int = 1,
    patience: int = 10,
    ibs_grid_size: int = None,
    save_plots: bool = False,
):

    patience_str = f", patience={patience}" if patience else ""
    logger.info(
        f"K-Fold CV XGBoost IBS  started "
        f"(k={n_folds}, loss={xgb_loss}, max_rounds={max_n_rounds}, step={round_step}{patience_str})"
    )
    df = load_processed_dataset(dataset_name, method_name="xgboost", fix_split=False)
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

    method_config_path = repo_root / 'Model_Training' / 'method_configs' / 'xgboost.yaml'
    if not method_config_path.exists():
        print(f"[WARNING] Method config not found: {method_config_path}")
    with open(method_config_path, "r") as f:
        method_config = yaml.safe_load(f)

    if time_unit is None:
        time_unit = method_config.get('TIME_UNIT', 'time units')

    grid_dict = method_config.get("hyperparameter_grid", {}).get(xgb_loss, {})

    grid_dict_no_rounds = {k: v for k, v in grid_dict.items() if k != 'num_boost_round'}
    keys = list(grid_dict_no_rounds.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid_dict_no_rounds[k] for k in keys])]
    n_checkpoints = len(range(round_step, max_n_rounds + 1, round_step))
    logger.info(
        f"HP grid: {len(combos)} combos x {n_checkpoints} checkpoints "
        f"= {len(combos) * n_checkpoints} total evaluations "
        f"(every {round_step} rounds up to {max_n_rounds})"
    )

    plots_base = EXPERIMENT_RESULTS / 'plots' / 'xgboost' / 'ibs'

    start_time = time.perf_counter()
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []
    fold_indices = {}
    failed_combinations = []
    failed_folds = []

    FOLD_TIME_BUDGET_S = fold_time_budget_h * 3600
    budget_exceeded = False
    all_sweep_data = []

    for fold, (trainfull_idx, test_idx) in enumerate(kfold.split(X_all_np, y_event_all), start=1):
        logger.info(f"\n{'='*60}\nFold {fold}/{n_folds} | XGBoost ({xgb_loss}) | IBS [EARLY-STOP] | {dataset_name}\n{'='*60}")

        trainfull_idx = np.asarray(trainfull_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)

        train_rel_idx, val_rel_idx = train_test_split(
            np.arange(len(trainfull_idx)),
            test_size=0.2,
            random_state=seed + fold,
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

        X_train.columns = [str(c).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') for c in X_train.columns]
        X_val.columns = [str(c).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') for c in X_val.columns]
        X_test.columns = [str(c).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') for c in X_test.columns]

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

        if xgb_loss == "cox":
            objective = "survival:cox"
        elif xgb_loss == "aft":
            objective = "survival:aft"

        dtrain = convert_data_to_xgb_format(X_train, y_train, objective)
        dval = convert_data_to_xgb_format(X_val, y_val, objective)
        dtest = convert_data_to_xgb_format(X_test, y_test, objective)

        time_grid = get_time_grid(df, y_train, y_val)
        time_grid = subsample_time_grid(time_grid, ibs_grid_size, logger)
        if time_grid is None or len(time_grid) == 0:
            logger.warning(f"Fold {fold}: empty time grid, skipping")
            continue

        # Post-training early stopping
        (best_model, best_hp_idx, best_n_rounds, best_val_ibs,
         sweep_log, fold_failed, budget_exceeded, fold_hp_search_seconds,
        ) = select_best_checkpoint(
            combos, objective, xgb_loss,
            dtrain, dval, y_train, time_grid,
            Y_val, D_val, Y_train, D_train,
            max_n_rounds, round_step, seed, fold,
            FOLD_TIME_BUDGET_S, logger, patience=patience,
        )
        failed_combinations.extend(fold_failed)

        if budget_exceeded:
            logger.warning("Aborting remaining folds due to time budget.")
            break
        if best_hp_idx is None or best_model is None:
            logger.warning(f"Fold {fold}: no valid HP/round found")
            continue

        best_params = combos[best_hp_idx]
        logger.info(
            f"Fold {fold} BEST: HP={best_params}, n_rounds={best_n_rounds}, "
            f"val_IBS={best_val_ibs:.6f}"
        )

        # Test evaluation using best checkpoint (no refit)
        time_grid_test = get_time_grid(df, y_train, y_test)
        time_grid_test = subsample_time_grid(time_grid_test, ibs_grid_size, logger)

        n_test = len(Y_test)
        t_test = time_grid_test.astype(float)

        # Reconstruct trainfull (train + val) for KM baseline and IPCW weights,
        y_trainfull_km = np.concatenate([y_train, y_val])
        Y_trainfull_km, D_trainfull_km = split_y(y_trainfull_km)

        random_ibs, km_ibs, km_mae_po = compute_baselines(
            Y_test, D_test, Y_trainfull_km, D_trainfull_km, t_test, n_test, logger, fold
        )

        t_infer_start = time.perf_counter()
        best_sliced = best_model[:best_n_rounds]

        if xgb_loss == "aft":
            preds_test_mu_log = best_sliced.predict(dtest, output_margin=True)
            dist = best_params.get("aft_loss_distribution", "normal")
            scale = best_params.get("aft_loss_distribution_scale", 1.0)
            native_times_test, native_surv_test = get_native_surv_xgb_aft(
                preds_test_mu_log, y_train, dist=dist, scale=scale,
            )
        elif xgb_loss == "cox":
            lin_pred_train = best_sliced.predict(dtrain, output_margin=True)
            lin_pred_test = best_sliced.predict(dtest, output_margin=True)
            native_times_test, native_surv_test = get_native_surv_xgb_cox(
                y_train, lin_pred_train, lin_pred_test,
            )

        surv_on_eval_grid = interpolate_survival_to_grid(
            native_times_test, native_surv_test, time_grid_test,
        )
        surv_on_eval_grid = np.clip(surv_on_eval_grid, 1e-6, 1.0 - 1e-6)
        t_infer_end = time.perf_counter()
        fold_inference_seconds = t_infer_end - t_infer_start

        n_combos_evaluated = len(set(s["hp_idx"] for s in sweep_log))
        logger.info(
            f"Fold {fold} TIMING: HP search={fold_hp_search_seconds:.2f}s "
            f"({n_combos_evaluated} combos, {len(sweep_log)} checkpoints evaluated), "
            f"inference={fold_inference_seconds:.4f}s"
        )

        test_ibs, test_c, x_cal_test, p_value, bin_stats, mae_po_score = evaluate_on_test(
            surv_on_eval_grid, time_grid_test, Y_test, D_test, Y_trainfull_km, D_trainfull_km
        )

        xgb_model_name = f"xgboost_{xgb_loss}_earlystop"

        if save_plots:
            calib_path = save_calibration_histogram(
                bin_stats, dataset_name, xgb_model_name, fold,
                save_dir=plots_base / 'calibration_histogram' / xgb_model_name,
                metric_label="ibs",
            )
            logger.info(f"Saved calibration plot: {calib_path}")

            if xgb_loss == "aft":
                risk_scores = -preds_test_mu_log
            else:
                risk_scores = lin_pred_test

            km_path = plot_risk_stratified_km(
                risk_scores, Y_test, D_test, Y_train, D_train,
                dataset_name, xgb_model_name, fold,
                save_dir=plots_base / 'km_curves' / xgb_model_name,
                metric_label="ibs",
            )
            logger.info(f"Saved KM curve plot: {km_path}")

            native_style = "step" if xgb_loss == "cox" else "line"
            surv_path = plot_native_vs_interpolated(
                native_times_test, native_surv_test, time_grid_test, surv_on_eval_grid,
                test_idx, dataset_name, xgb_model_name, fold, seed,
                save_dir=plots_base / 'individual_surv' / xgb_model_name,
                metric_label="ibs",
                native_style=native_style,
            )
            logger.info(f"Saved native vs interpolated survival curves: {surv_path}")
            plt.close('all')

        logger.info(
            f"Fold {fold} TEST: IBS={test_ibs:.6f}, C-index={test_c:.6f}, "
            f"MAE-PO={mae_po_score:.4f}, x_cal={x_cal_test:.6f}, p={p_value:.6g}"
        )

        all_sweep_data.append({
            "fold": fold,
            "sweep_log": [{"hp_idx": s["hp_idx"], "n_rounds": s["n_rounds"],
                           "val_metric": s["val_ibs"]} for s in sweep_log],
            "best_hp_idx": best_hp_idx,
            "best_n_rounds": best_n_rounds,
            "best_val_metric": best_val_ibs,
        })

        fold_results.append({
            "fold": fold,
            "best_params": best_params,
            "best_n_rounds": best_n_rounds,
            "n_dropped_val": n_dropped_val,
            "n_dropped_test": n_dropped_test,
            "fold_hp_search_seconds": fold_hp_search_seconds,
            "fold_inference_seconds": fold_inference_seconds,
            "val_ibs": best_val_ibs,
            "test_ibs": test_ibs,
            "test_c": test_c,
            "test_x_cal": x_cal_test,
            "test_p": float(p_value),
            "mae_po": float(mae_po_score),
            "km_train_mae_po": float(km_mae_po),
            "random_ibs": random_ibs,
            "km_train_ibs": km_ibs,
            "sweep_log": sweep_log,
        })

        del best_model
        gc.collect()

    elapsed = time.perf_counter() - start_time

    (mean_test_ibs, std_test_ibs, mean_test_c, std_test_c,
     mean_random_ibs, mean_km_ibs, mean_mae_po, std_mae_po, mean_km_mae_po) = log_cv_summary(
        fold_results, logger, dataset_name, t_min, t_max, time_unit, n_folds, elapsed, n=len(df),
        label="[EARLY-STOP]"
    )

    xgb_model_name = f"xgboost_{xgb_loss}_earlystop"
    if save_plots and all_sweep_data:
        sweep_path = plot_xgb_sweep_curves(
            all_sweep_data, dataset_name, xgb_model_name,
            save_dir=plots_base / 'training_curves' / xgb_model_name,
            metric_name="IBS", higher_is_better=False,
        )
        if sweep_path:
            logger.info(f"Saved sweep curves plot: {sweep_path}")

    results_dir = EXPERIMENT_RESULTS / 'results' / 'xgboost' / 'ibs'
    os.makedirs(results_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = results_dir / f"kfold_xgboost_{xgb_loss}_{dataset_name}_{ts}_ibs_earlystop.json"

    serializable_folds = []
    for r in fold_results:
        r_copy = dict(r)
        r_copy.pop("sweep_log", None)
        serializable_folds.append(r_copy)

    details = {
        "dataset": dataset_name,
        "xgb_loss": xgb_loss,
        "seed": seed,
        "n_folds": n_folds,
        "cindex_method": "pycox_adj_antolini",
        "interpolation": "linear (standardized)",
        "optimization_metric": "IBS",
        "early_stopping": True,
        "max_n_rounds": max_n_rounds,
        "round_step": round_step,
        "patience": patience,
        "no_refit": True,
        "mean_test_ibs": mean_test_ibs,
        "std_test_ibs": std_test_ibs,
        "mean_test_c": mean_test_c,
        "std_test_c": std_test_c,
        "mean_test_mae_po": mean_mae_po,
        "std_test_mae_po": std_mae_po,
        "mean_random_ibs": mean_random_ibs,
        "mean_km_ibs": mean_km_ibs,
        "mean_km_mae_po": mean_km_mae_po,
        "search_space": grid_dict_no_rounds,
        "hp_combinations": [{"hp_index": i, "params": c} for i, c in enumerate(combos)],
        "num_hp_combinations": len(combos),
        "num_checkpoints_per_combo": n_checkpoints,
        "total_evaluations": len(combos) * n_checkpoints,
        "elapsed_seconds": elapsed,
        "fold_results": serializable_folds,
        "fold_indices": fold_indices if len(df) <= 10_000 else "omitted (n > 10k)",
        "failed_combinations": failed_combinations,
        "num_failed": len(failed_combinations),
        "failed_folds": failed_folds,
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

    parser = argparse.ArgumentParser(
        description="XGBoost IBS with post-training early stopping"
    )
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--fold_time_budget_h', type=float, default=3.0)
    parser.add_argument('--max_n_rounds', type=int, default=100,
                        help='Max boosting rounds per HP combo (default: 100)')
    parser.add_argument('--round_step', type=int, default=1,
                        help='Checkpoint evaluation interval (default: 1)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Stop evaluating an HP combo after this many checkpoints '
                             'without improvement (default: 10)')
    parser.add_argument('--losses', type=str, nargs='+', default=['cox', 'aft'],
                        choices=['cox', 'aft'])
    parser.add_argument('--ibs_grid_size', type=int, default=None,
                        help='Subsample IBS time grid to this many evenly-spaced points (default: None = no subsampling)')
    parser.add_argument('--save_plots', action='store_true', default=False)
    args = parser.parse_args()

    seed = 42

    for dataset_name in args.datasets:
        for xgb_loss in args.losses:
            logger.info(f"\nProcessing {dataset_name} with {xgb_loss} loss [EARLY-STOP]")

            logs_dir = EXPERIMENT_RESULTS / 'logs' / 'xgboost' / 'ibs'
            os.makedirs(logs_dir, exist_ok=True)

            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(message)s",
                handlers=[
                    logging.FileHandler(
                        logs_dir / f"xgboost_{xgb_loss}_kfold_ibs_{dataset_name}_earlystop.log",
                        mode="w",
                    ),
                    logging.StreamHandler()
                ],
                force=True
            )

            try:
                optimization_xgboost_kfold(
                    dataset_name=dataset_name,
                    seed=seed,
                    n_folds=args.n_folds,
                    xgb_loss=xgb_loss,
                    fold_time_budget_h=args.fold_time_budget_h,
                    max_n_rounds=args.max_n_rounds,
                    round_step=args.round_step,
                    patience=args.patience,
                    ibs_grid_size=args.ibs_grid_size,
                    save_plots=args.save_plots,
                )
            except Exception as e:
                logger.error(f"Failed: {e}")
            finally:
                gc.collect()
