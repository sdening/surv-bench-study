'''training utils for trainign survtrace model.
'''
from collections import defaultdict
import pdb
import os
import math
import numpy as np
import torch
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_
from torch import optim

from .losses import NLLPCHazardLoss
 
from copy import deepcopy #manual edit
from pycox.evaluation import EvalSurv #manual edit

def _better(new, best, minimize: bool): #manual edit
    return (new < best) if minimize else (new > best)


import numpy as np
import pandas as pd
from SurvivalEVAL.Evaluator import SurvivalEvaluator

def _resample_surv_df_to_grid(
    surv_df: pd.DataFrame,
    time_grid: np.ndarray,
) -> np.ndarray:
    """
    surv_df: pandas DataFrame, shape (n_model_times, n_samples)
             index = model time coordinates (raw scale), increasing
             columns = samples
    time_grid: 1D array of desired times (raw scale), increasing

    Returns:
        pred_survs: np.ndarray shape (n_samples, len(time_grid))
                   survival probs on time_grid
    """
    if not isinstance(surv_df, pd.DataFrame):
        raise TypeError(f"surv_df must be a pd.DataFrame, got {type(surv_df)}")

    # model time coordinates
    t_src = np.asarray(surv_df.index, dtype=float)
    if t_src.ndim != 1 or len(t_src) < 2:
        raise ValueError("surv_df.index must be a 1D increasing time grid with >=2 points")

    # desired grid
    t_tgt = np.asarray(time_grid, dtype=float).copy()
    if t_tgt.ndim != 1 or len(t_tgt) < 2:
        raise ValueError("time_grid must be a 1D increasing grid with >=2 points")

    # ensure strictly increasing (avoid interpolation weirdness)
    # (if you *know* your grid is already fine, you can remove this)
    t_tgt = np.unique(t_tgt)
    t_tgt.sort()

    # values: (n_src_times, n_samples)
    S_src = surv_df.to_numpy(dtype=float, copy=False)

    # Safety: Survival should be within [0,1]
    # (don’t hard-clip aggressively unless you want to; this is mild)
    S_src = np.clip(S_src, 0.0, 1.0)

    n_src, n_samples = S_src.shape
    n_tgt = len(t_tgt)

    # Interpolate each sample curve onto target grid.
    # For t < first model time -> survival ~ 1
    # For t > last model time  -> hold last value
    pred = np.empty((n_samples, n_tgt), dtype=float)
    for j in range(n_samples):
        y = S_src[:, j]
        pred[j, :] = np.interp(t_tgt, t_src, y, left=1.0, right=float(y[-1]))

    return pred


def compute_ibs_survtrace(
    *,
    surv_df: pd.DataFrame,
    time_grid: np.ndarray,
    y_val_raw,
    y_train_raw,
    ipcw_weighted: bool = True,
) -> float:
    """
    Compatible with your TrainerSurvTrace hook signature:

        compute_ibs_fn(
            surv_df=surv_val_df,
            time_grid=time_grid_val,
            y_val_raw=(Y_val_raw, D_val_raw),
            y_train_raw=(Y_train_raw, D_train_raw),
        )

    Returns:
        IBS (float)
    """
    if y_val_raw is None or y_train_raw is None:
        raise ValueError("Need y_val_raw and y_train_raw for IBS (IPCW).")

    Y_val, D_val = y_val_raw
    Y_tr, D_tr = y_train_raw

    Y_val = np.asarray(Y_val, dtype=float)
    D_val = np.asarray(D_val, dtype=int)
    Y_tr  = np.asarray(Y_tr, dtype=float)
    D_tr  = np.asarray(D_tr, dtype=int)

    # Resample surv_df to (n_val, n_times)
    pred_survs = _resample_surv_df_to_grid(surv_df, time_grid)

    # IMPORTANT: time_coordinates must match pred_survs second dim
    t = np.unique(np.asarray(time_grid, dtype=float))
    t.sort()

    if pred_survs.shape != (len(Y_val), len(t)):
        raise ValueError(
            f"Shape mismatch after resampling: pred_survs={pred_survs.shape}, "
            f"expected ({len(Y_val)}, {len(t)}). "
            f"Check that surv_df has one column per val sample and that y_val_raw matches."
        )

    ev = SurvivalEvaluator(
        pred_survs=pred_survs,
        time_coordinates=t,
        event_times=Y_val,
        event_indicators=D_val,
        train_event_times=Y_tr,
        train_event_indicators=D_tr,
    )

    ibs = float(ev.integrated_brier_score(
        num_points=None,
        target_times=t,
        IPCW_weighted=ipcw_weighted,
        integration_method="trapz",
        draw_figure=False,
    ))
    return ibs


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, name='checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), name)
        self.val_loss_min = val_loss

def pad_col(input, val=0, where='end'):
    """Addes a column of `val` at the start of end of `input`."""
    if len(input.shape) != 2:
        raise ValueError(f"Only works for `phi` tensor that is 2-D.")
    pad = torch.zeros_like(input[:, :1])
    if val != 0:
        pad = pad + val
    if where == 'end':
        return torch.cat([input, pad], dim=1)
    elif where == 'start':
        return torch.cat([pad, input], dim=1)
    raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")

    
############################
# optimizer #
############################

def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + math.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

SCHEDULES = {
    'warmup_cosine':warmup_cosine,
    'warmup_constant':warmup_constant,
    'warmup_linear':warmup_linear,
}

class BERTAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix (and no ).
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay_rate: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """
    def __init__(self, params, lr, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay_rate=0.01,
                 max_grad_norm=1.0):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= warmup < 1.0 and not warmup == -1:
            raise ValueError("Invalid warmup: {} - should be in [0.0, 1.0[ or -1".format(warmup))
        if not 0.0 <= b1 < 1.0:
            raise ValueError("Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1))
        if not 0.0 <= b2 < 1.0:
            raise ValueError("Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2))
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        defaults = dict(lr=lr, schedule=schedule, warmup=warmup, t_total=t_total,
                        b1=b1, b2=b2, e=e, weight_decay_rate=weight_decay_rate,
                        max_grad_norm=max_grad_norm)
        super(BERTAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        print("l_total=",len(self.param_groups))
        for group in self.param_groups:
            print("l_p=",len(group['params']))
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                pdb.set_trace()
                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']
                lr.append(lr_scheduled)
        return lr

    def to(self, device):
        """ Move the optimizer state to a specified device"""
        for state in self.state.values():
            state['exp_avg'].to(device)
            state['exp_avg_sq'].to(device)

    def initialize_step(self, initial_step):
        """Initialize state with a defined step (but we don't have stored averaged).
        Arguments:
            initial_step (int): Initial step number.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # State initialization
                state['step'] = initial_step
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Add grad clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group['e'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want ot decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group['weight_decay_rate'] > 0.0:
                    update += group['weight_decay_rate'] * p.data

                if group['t_total'] != -1:
                    schedule_fct = SCHEDULES[group['schedule']]
                    lr_scheduled = group['lr'] * schedule_fct(state['step']/group['t_total'], group['warmup'])
                else:
                    lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss


############################
# trainer #
############################

class TrainerSurvTrace:
    def __init__(self, model, device, metrics=None):
        '''metrics must start from NLLPCHazardLoss, then be others
        '''
        self.model = model
        self.device = device
        if metrics is None:
            self.metrics = [NLLPCHazardLoss(),]

        self.train_logs = defaultdict(list)
        self.get_target = lambda df: (df['duration'].values, df['event'].values)
        self.use_gpu = True if torch.cuda.is_available() else False
        if self.use_gpu:
            print('use pytorch-cuda for training.')
            self.model.cuda()
            self.model.use_gpu = True
        else:
            print('GPU not found! will use cpu for training!')
        self.early_stopping = None
        ckpt_dir = os.path.dirname(model.config['checkpoint'])
        self.ckpt = model.config['checkpoint']
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

    def train_single_event(
        self,
        train_set,
        val_set=None,
        batch_size=64,
        epochs=100,
        learning_rate=1e-3,
        weight_decay=0,
        val_batch_size=None,
        eval_every=10,
        selection_metric="antolini",   # "antolini" or "ibs"
        minimize_metric=False,         # True if selection_metric=="ibs"
        time_grid_val=None,            # np.array of times (raw scale) for IBS
        y_val_raw=None,                # tuple (Y_val_raw, D_val_raw)
        y_train_raw=None,              # tuple (Y_train_raw, D_train_raw) for IBS/IPCW
        compute_ibs_fn=None,           # function hook to use your SurvivalEvaluator IBS
        **kwargs,
    ):
        #local survtrace trainer -- adapted!
        df_train, df_y_train = train_set
        durations_train, events_train = self.get_target(df_y_train)

        if val_set is not None:
            df_val, df_y_val = val_set
            #durations_val, events_val = self.get_target(df_y_val)
            tensor_val = torch.tensor(val_set[0].values)
            tensor_y_val = torch.tensor(val_set[1].values)
        
            if self.use_gpu:
                tensor_val = tensor_val.cuda()
                tensor_y_val = tensor_y_val.cuda()

        # Use Adam optimizer like DeepHit instead of BERTAdam
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )

        #if val_set is not None: #manual edit comment 
        #    # take early stopping
        #    self.early_stopping = EarlyStopping(patience=self.model.config['early_stop_patience']) #manual edit comment 

        num_train_batch = int(np.ceil(len(df_y_train) / batch_size))
        train_loss_list, val_loss_list = [], []
        metric_history = [] #manual edit

        # best checkpoint based on metric
        best_metric = np.inf if minimize_metric else -np.inf #manual edit
        best_epoch = None #manual edit
        best_state = None #manual edit
        val_loss_value = None #manual edit
        
        for epoch in range(epochs):
            epoch_loss = 0
            self.model.train()
            df_train_shuf = train_set[0].sample(frac=1)
            df_y_train_shuf = train_set[1].loc[df_train_shuf.index]

            tensor_train = torch.tensor(df_train_shuf.values)
            tensor_y_train = torch.tensor(df_y_train_shuf.values)
            if self.use_gpu:
                tensor_train = tensor_train.cuda()
                tensor_y_train = tensor_y_train.cuda()

            for batch_idx in range(num_train_batch):
                optimizer.zero_grad()

                batch_train = tensor_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                batch_y_train = tensor_y_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                
                batch_x_cat = batch_train[:, :self.model.config.num_categorical_feature].long()
                batch_x_num = batch_train[:, self.model.config.num_categorical_feature:].float()


                phi = self.model(input_ids=batch_x_cat, input_nums=batch_x_num)

                
                #manual edit temp fix
                #interval_frac = torch.full_like(batch_y_train[:,0].float(), 0.5)
                batch_loss = self.metrics[0](phi[1], batch_y_train[:,0].long(), batch_y_train[:,1].long(), batch_y_train[:,2].float(), reduction="mean")
                #batch_loss = self.metrics[0](phi[1], batch_y_train[:,0].long(), 
                #           batch_y_train[:,1].long(), interval_frac)

                #else:
                #    raise NotImplementedError

                batch_loss.backward()
                #for name, param in self.model.named_parameters():
                #    if param.grad is not None:
                #        print(f"{name} grad norm: {param.grad.norm():.6f}")
                optimizer.step()

                epoch_loss += float(batch_loss.item()) #manual edit added float

            train_loss_list.append(epoch_loss / (batch_idx+1))

            if val_set is not None:
                self.model.eval()
                with torch.no_grad():
                    phi_val = self.model.predict(tensor_val, val_batch_size)
                    #print("Hazard logits phi[1] stats:")
                    #print(f"  mean: {phi_val[1].mean():.4f}, std: {phi_val[1].std():.4f}")
                    #print(f"  min: {phi_val[1].min():.4f}, max: {phi_val[1].max():.4f}")
                    #print(f"  unique values: {torch.unique(phi_val[1]).numel()} / {phi_val[1].numel()}")
                
                val_loss = self.metrics[0](phi_val, tensor_y_val[:,0].long(), tensor_y_val[:,1].long(), tensor_y_val[:,2].float())
                #print("[Train-{}]: {}".format(epoch, epoch_loss))
                #print("[Val-{}]: {}".format(epoch, val_loss.item()))
                #val_loss_list.append(val_loss.item()) #manual edit comment
                #self.early_stopping(val_loss.item(), self.model, name=self.ckpt) #manual edit comment
                #if self.early_stopping.early_stop: #manual edit comment
                #    print(f"early stops at epoch {epoch+1}") #manual edit comment
                #    # load best checkpoint #manual edit comment
                #    self.model.load_state_dict(torch.load(self.ckpt)) #manual edit comment
                #    return train_loss_list, val_loss_list #manual edit comment
                val_loss_value = float(val_loss.item()) #manual edit
                val_loss_list.append(val_loss_value) #manual edit

            ## eval every 10th epoch IBS and cindex logic! # manual edit
            do_eval = (val_set is not None) and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1)
            if do_eval:
                surv_val_df = self.model.predict_surv_df(df_val, batch_size=val_batch_size)

                if selection_metric == "antolini":
                    if y_val_raw is None:
                        raise ValueError("Need y_val_raw=(Y_val_raw, D_val_raw) for Antolini C-index.")
                    Yv, Dv = y_val_raw



                    c = float(EvalSurv(surv_val_df, Yv, Dv).concordance_td("antolini"))
                    metric_value = c
                    print(f"    [epoch {epoch+1}] val_cindex_antolini={c:.4f}  val_loss={val_loss_value:.4f}")

                elif selection_metric == "ibs":
                    if compute_ibs_fn is None:
                        raise ValueError("Provide compute_ibs_fn(...) to compute IBS the same way as your pipeline.")
                    metric_value = float(compute_ibs_fn(
                        surv_df=surv_val_df,
                        time_grid=time_grid_val,
                        y_val_raw=y_val_raw,
                        y_train_raw=y_train_raw
                    ))
                else:
                    raise ValueError(f"Unknown selection_metric: {selection_metric}")

                metric_history.append((epoch+1, metric_value))

                # checkpoint on metric
                is_better = _better(metric_value, best_metric, minimize_metric)
                if _better(metric_value, best_metric, minimize_metric):
                    best_metric = metric_value
                    best_epoch = epoch + 1
                    #best_state = deepcopy(self.model.state_dict())
                    # optional: persist to disk
                    #torch.save(best_state, self.ckpt)
                    torch.save(self.model.state_dict(), self.ckpt)

        # load best checkpoint at end
        #if best_state is not None:
        #    self.model.load_state_dict(best_state)
        # load best checkpoint at end
        if best_epoch is not None and os.path.exists(self.ckpt):
            self.model.load_state_dict(torch.load(self.ckpt, map_location=self.device))


        return {
            "best_val_cindex": float(best_metric),   
            "best_epoch": int(best_epoch),
            "best_state_dict": best_state,          
            "history": {
                "epochs": list(range(1, len(train_loss_list) + 1)),
                "train_loss": train_loss_list,
                "val_loss": val_loss_list,
                "val_cindex": metric_history,    
            },
        }

        #return train_loss_list, val_loss_list


    def fit(self, 
        train_set,
        val_set=None,
        batch_size=64,
        epochs=100,
        learning_rate=1e-3,
        weight_decay=0,
        val_batch_size=None,
        **kwargs,
        ):
        '''fit on the train_set, validate on val_set for early stop
        params should have the following terms:
        batch_size,
        epochs,
        optimizer,
        metric,
        '''
        if self.model.config.num_event == 1:
            return self.train_single_event(
                    train_set=train_set,
                    val_set=val_set,
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    val_batch_size=val_batch_size,
                    **kwargs,
            )
        
        elif self.model.config.num_event > 1:
            return self.train_multi_event(
                    train_set=train_set,
                    val_set=val_set,
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    val_batch_size=val_batch_size,
                    **kwargs,
            )
        
        else:
            raise ValueError
