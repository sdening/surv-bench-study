import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sksurv.nonparametric import kaplan_meier_estimator


def plot_native_vs_interpolated(
    native_times,
    native_surv,
    eval_grid,
    surv_on_eval_grid,
    test_idx,
    dataset_name: str,
    model_name: str,
    fold: int,
    seed: int,
    save_dir: Path,
    metric_label: str = "ibs",
    n_curves: int = 5,
    native_style: str = "step",
    timegrid_method: str = None,
):
    """Side-by-side plot of native model output vs interpolated evaluation curves."""
    rng = np.random.default_rng(seed + fold)
    n_plot = min(n_curves, surv_on_eval_grid.shape[0])
    sample_indices = rng.choice(surv_on_eval_grid.shape[0], size=n_plot, replace=False)
    colors = plt.cm.tab10(np.linspace(0, 1, n_plot))

    fig, (ax_nat, ax_int) = plt.subplots(1, 2, figsize=(18, 6), sharey=True, sharex=True)

    for i, idx in enumerate(sample_indices):
        lbl = f"Test idx {test_idx[idx]}"
        if native_style == "step":
            ax_nat.step(native_times, native_surv[idx], where="post",
                        label=lbl, color=colors[i], linewidth=2)
        else:
            ax_nat.plot(native_times, native_surv[idx],
                        label=lbl, color=colors[i], linewidth=2,
                        marker=".", markersize=4)
        ax_int.plot(eval_grid, surv_on_eval_grid[idx],
                    label=lbl, color=colors[i], linewidth=2,
                    marker=".", markersize=3)

    eg_lo, eg_hi = float(eval_grid[0]), float(eval_grid[-1])
    for ax in (ax_nat, ax_int):
        ax.axvline(eg_lo, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axvline(eg_hi, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_nat.fill_betweenx([0, 1], eg_lo, eg_hi, alpha=0.07, color="blue",
                         label=f"Eval window [{eg_lo:.1f}, {eg_hi:.1f}]")

    for ax in (ax_nat, ax_int):
        ax.set_xlabel("Time")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    style_desc = "step function" if native_style == "step" else "discrete anchors"
    ax_nat.set_ylabel("Survival Probability")
    ax_nat.set_title(
        f"Native Model Output ({style_desc}, {len(native_times)} points)\n"
        f"{dataset_name} - {model_name} - Fold {fold}"
    )
    ax_int.set_title(
        f"After Standardized Linear Interpolation ({len(eval_grid)} eval-grid points)\n"
        f"{dataset_name} - {model_name} - Fold {fold}"
    )

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tg_suffix = f"_{timegrid_method}" if timegrid_method else ""
    fname = f"surv_{dataset_name}_{model_name.lower()}{tg_suffix}_{metric_label}_fold{fold}_{ts}.png"
    save_path = save_dir / fname
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_risk_stratified_km(
    risk_scores,
    Y_test,
    D_test,
    Y_train,
    D_train,
    dataset_name: str,
    model_name: str,
    fold: int,
    save_dir: Path,
    metric_label: str = "ibs",
    timegrid_method: str = None,
):
    """Risk-stratified KM curves (tertile groups) with training KM reference."""
    risk_tertiles = np.percentile(risk_scores, [33.3, 66.7])
    low_risk = risk_scores <= risk_tertiles[0]
    med_risk = (risk_scores > risk_tertiles[0]) & (risk_scores <= risk_tertiles[1])
    high_risk = risk_scores > risk_tertiles[1]

    fig, ax = plt.subplots(figsize=(10, 6))

    for mask, label, color in [
        (low_risk, "Low Risk", "green"),
        (med_risk, "Medium Risk", "orange"),
        (high_risk, "High Risk", "red"),
    ]:
        if np.sum(mask) > 0:
            t_km, s_km = kaplan_meier_estimator(D_test[mask].astype(bool), Y_test[mask])
            ax.step(t_km, s_km, where="post", label=label, color=color, linewidth=2)

    t_tr, s_tr = kaplan_meier_estimator(D_train.astype(bool), Y_train)
    ax.step(t_tr, s_tr, where="post", label="Train KM", color="black",
            linestyle="--", linewidth=1.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.set_title(f"Risk-Stratified KM Curves - {dataset_name} - Fold {fold}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tg_suffix = f"_{timegrid_method}" if timegrid_method else ""
    fname = f"km_{dataset_name}_{model_name.lower()}{tg_suffix}_{metric_label}_fold{fold}_{ts}.png"
    save_path = save_dir / fname
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def save_calibration_histogram(
    bin_stats,
    dataset_name: str,
    model_name: str,
    fold: int,
    save_dir: Path,
    metric_label: str = "ibs",
    timegrid_method: str = None,
):
    """Save the d-calibration histogram produced by SurvivalEVAL."""
    fig, _ax = bin_stats["histogram_plot"]
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tg_suffix = f"_{timegrid_method}" if timegrid_method else ""
    fname = f"hist_{dataset_name}_{model_name.lower()}{tg_suffix}_{metric_label}_fold{fold}_{ts}.png"
    save_path = save_dir / fname
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_xgb_sweep_curves(
    fold_sweep_data: list,
    dataset_name: str,
    model_name: str,
    save_dir: Path,
    metric_name: str = "IBS",
    higher_is_better: bool = False,
):
    """Plot XGBoost checkpoint sweep curves (analogous to DL training curves).
    """
    if not fold_sweep_data:
        return None

    n_folds = len(fold_sweep_data)
    n_cols = min(n_folds, 3)
    n_rows = (n_folds + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows),
                             squeeze=False)

    cmap = plt.cm.tab20

    for i, fdata in enumerate(fold_sweep_data):
        ax = axes[i // n_cols][i % n_cols]
        fold = fdata["fold"]
        sweep_log = fdata["sweep_log"]
        best_hp_idx = fdata["best_hp_idx"]
        best_n_rounds = fdata["best_n_rounds"]
        best_val = fdata["best_val_metric"]

        hp_indices = sorted(set(e["hp_idx"] for e in sweep_log))

        for hi in hp_indices:
            entries = [e for e in sweep_log if e["hp_idx"] == hi]
            rounds = [e["n_rounds"] for e in entries]
            vals = [e["val_metric"] for e in entries]
            is_best = (hi == best_hp_idx)
            color = cmap(hi % 20)
            ax.plot(rounds, vals,
                    color=color,
                    linewidth=2.0 if is_best else 0.7,
                    alpha=1.0 if is_best else 0.4,
                    label=f"HP {hi+1}" if is_best else None,
                    zorder=10 if is_best else 1)

        ax.plot(best_n_rounds, best_val, '*', color='red', markersize=14,
                zorder=20, label=f"Best: r={best_n_rounds}, {metric_name}={best_val:.4f}")

        ax.set_xlabel("Boosting Rounds")
        ax.set_ylabel(f"Val {metric_name}")
        ax.set_title(f"Fold {fold} ({len(hp_indices)} HP combos)")
        ax.legend(fontsize=7, loc="upper right" if not higher_is_better else "lower right")
        ax.grid(True, alpha=0.3)

    for i in range(n_folds, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].axis('off')

    fig.suptitle(
        f"XGBoost Checkpoint Sweep - {dataset_name} - {model_name}\n"
        f"Val {metric_name} vs. Boosting Rounds",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"sweep_curves_{dataset_name}_{model_name.lower()}_{metric_name.lower()}_{ts}.png"
    save_path = save_dir / fname
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_cv_splits(
    fold_indices: dict,
    n_samples: int,
    n_folds: int,
    dataset_name: str,
    model_name: str,
    save_dir: Path,
    metric_label: str = "ibs",
):
    """Heatmap of K-fold train/val/test assignment per sample."""
    split_matrix = np.zeros((n_samples, n_folds), dtype=int)

    for fold_key, indices in fold_indices.items():
        fold_idx = int(fold_key.split("_")[1]) - 1
        split_matrix[np.array(indices["train_indices"]), fold_idx] = 0
        split_matrix[np.array(indices["val_indices"]), fold_idx] = 1
        split_matrix[np.array(indices["test_indices"]), fold_idx] = 2

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(split_matrix.T)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Fold")
    ax.set_title(
        f"K-Fold CV Split Visualization ({n_folds} folds)\n"
        "64% train, 16% val, 20% test per fold"
    )
    ax.set_yticks(range(n_folds))
    ax.set_yticklabels([f"Fold {i+1}" for i in range(n_folds)])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["Train", "Val", "Test"])

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"kfold_splits_{dataset_name}_{n_folds}folds_{model_name.lower()}_{metric_label}_{ts}.png"
    save_path = save_dir / fname
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path
