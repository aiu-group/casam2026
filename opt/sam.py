# source from https://github.com/davda54/sam/blob/main/sam.py

import torch
import torch.nn as nn
from collections import defaultdict, OrderedDict
from typing import Callable, Dict


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.base_rho = rho
        self.rho = rho
        self.grad_norm = 1.0
        self.outer_grad_norm = 1.0
        # self.defaults.update(self.base_optimizer.defaults)
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.base_optimizer.param_groups:
                group.setdefault(name, default)

    @torch.no_grad()
    def perturb_weight(self, zero_grad=True):
        self.grad_norm = self._grad_norm()
        for group in self.param_groups:
            # self.rho = self.base_rho * min(max(self.grad_norm**0.5, 1.0), 2.0)
            scale = self.rho / (self.grad_norm + 1e-6)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * scale * p.grad
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    def step(self):
        # self.outer_grad_norm = self._grad_norm()
        # scale = self.grad_norm / self.outer_grad_norm
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # p.grad.mul_(scale)
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm.item()

    def load_state_dict(self, state_dict):
        # add this line for Adam optimizer
        for group in state_dict["state"]:
            state_dict["state"][group] = {}
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
