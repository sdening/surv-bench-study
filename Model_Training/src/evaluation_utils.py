import numpy as np
import math
from scipy.special import erf
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

def breslow_fcn(times, events, lp):
    times = np.asarray(times)
    events = np.asarray(events).astype(bool)
    lp = np.asarray(lp)
    event_times = np.sort(np.unique(times[events]))
    exp_lp = np.exp(lp)
    H0_increments = []
    for t in event_times:
        d_j = np.sum(events & (times == t))
        denom = np.sum(exp_lp[times >= t])
        H0_increments.append(0.0 if denom == 0 else d_j / denom)
    H0 = np.cumsum(H0_increments)
    return event_times, H0

def surv_df_from_aft_mu(mu_log, time_grid, dist="normal", scale=1.0):
    """
    Map AFT output to S(t) for dist in ("normal", "logistic", "extreme")
    """
    mu = np.asarray(mu_log, float)[:, None]
    t  = np.asarray(time_grid, float)[None, :]
    t_log = np.log(np.clip(t, 1e-12, None))

    z = (t_log - mu) / scale

    if dist == "normal":
        #S = 0.5 * (1.0 - np.erf(z / np.sqrt(2.0))) #erf error.... -> use math library
        S = 0.5 * (1.0 - np.vectorize(math.erf)(z / np.sqrt(2.0)))
        #S = 0.5 * (1.0 - erf(z / np.sqrt(2.0)))
    elif dist == "logistic":
        F = 1.0 / (1.0 + np.exp(-z))
        S = 1.0 - F
    elif dist == "extreme":
        S = np.exp(-np.exp(z))
    else:
        raise ValueError("AFT distribution unknown")
    
    return S



def discretize_time_deephit(num_time_steps, D_train_np, Y_train_np, D_val_np=None, Y_val_np=None):
    """Discretize time for DeepHit using quantile bins or unique event times."""
    if num_time_steps == 0:
        mask = (D_train_np == 1)
        event_times = np.unique(Y_train_np[mask])
        if len(event_times) == 0:
            raise RuntimeError("No event times in TRAIN")
        cuts = event_times.astype(float)
        max_obs = Y_train_np.max()
        if Y_val_np is not None:
            max_obs = max(max_obs, Y_val_np.max())
        if max_obs > cuts.max():
            cuts = np.concatenate([cuts, np.array([max_obs], dtype=float)])
        min_obs = Y_train_np.min()
        if Y_val_np is not None:
            min_obs = min(min_obs, Y_val_np.min())
        if min_obs < cuts.min():
            cuts = np.concatenate([np.array([min_obs], dtype=float), cuts])
        label_transform = LabTransDiscreteTime(cuts)
    else:
        label_transform = LabTransDiscreteTime(num_time_steps, scheme='quantiles')
    
    Y_train_discrete_np, D_train_discrete_np = label_transform.fit_transform(Y_train_np, D_train_np)
    Y_val_discrete_np, D_val_discrete_np = label_transform.transform(Y_val_np, D_val_np)
    time_grid_train_np = label_transform.cuts
    output_num_time_steps = len(time_grid_train_np)
    
    return (time_grid_train_np, output_num_time_steps, Y_train_discrete_np, 
            D_train_discrete_np, Y_val_discrete_np, D_val_discrete_np)