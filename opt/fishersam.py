"""
to ease the usage in conjunction with various base optimizers,
we slightly modified the original version of SAM optimizer in Pytorch (https://github.com/davda54/sam),
and used the style of Lookahead optimizer implemented by Timm library
(https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/lookahead.py).
"""

import torch
import torch.nn as nn
from collections import defaultdict


class FisherSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho)
        super(FisherSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        # self.defaults.update(self.base_optimizer.defaults)
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.base_optimizer.param_groups:
                group.setdefault(name, default)

    @torch.no_grad()
    def perturb_weight(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                # note that according to (Kim et al., 2022), we use 1/(1+f) to approximate the reciprocal of f
                e_w = p.grad / (1.0 + p.grad**2) * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    (p.grad / (1.0 + p.grad**2) ** 0.5).norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        # add this line for Adam optimizer
        for group in state_dict["state"]:
            state_dict["state"][group] = {}
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
