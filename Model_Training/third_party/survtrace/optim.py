"""BERTAdam optimizer -- matches survhive's survtrace training behavior.

Key differences from standard Adam:
- Per-parameter gradient clipping (max_grad_norm=1.0)
- Decoupled weight decay (AdamW-style)
- No bias correction on moment estimates
"""
import torch
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_


class BERTAdam(Optimizer):
    """BERT-style Adam with per-param grad clipping and decoupled weight decay.

    Args:
        params: iterable of parameters or param groups
        lr: learning rate (default 1e-3)
        b1: beta1 for Adam (default 0.9)
        b2: beta2 for Adam (default 0.999)
        e: epsilon (default 1e-6)
        weight_decay_rate: decoupled weight decay (default 0.01)
        max_grad_norm: per-parameter gradient clipping norm (default 1.0, -1 to disable)
    """

    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, e=1e-6,
                 weight_decay_rate=0.01, max_grad_norm=1.0):
        defaults = dict(lr=lr, b1=b1, b2=b2, e=e,
                        weight_decay_rate=weight_decay_rate,
                        max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['next_m'] = torch.zeros_like(p)
                    state['next_v'] = torch.zeros_like(p)

                # Per-parameter gradient clipping
                if group['max_grad_norm'] > 0:
                    clip_grad_norm_(p, group['max_grad_norm'])

                next_m, next_v = state['next_m'], state['next_v']
                beta1, beta2 = group['b1'], group['b2']

                # Update biased first and second moment estimates
                next_m.mul_(beta1).add_(grad, alpha=1 - beta1)
                next_v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                update = next_m / (next_v.sqrt() + group['e'])

                # Decoupled weight decay (AdamW-style)
                if group['weight_decay_rate'] > 0.0:
                    update.add_(p, alpha=group['weight_decay_rate'])

                p.add_(update, alpha=-group['lr'])
                state['step'] += 1

        return loss
