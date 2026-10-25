import math
import torch
import torch.optim
from collections import defaultdict

class VASSO(torch.optim.Optimizer):
    def __init__(self, base_optimizer, rho, theta) -> None:
        # NOTE super().__init__() not called on purpose
        
        defaults = dict(rho=rho, theta=theta)
            
        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.base_optimizer.param_groups:
                group.setdefault(name, default)
                
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for group in self.param_groups:
            theta = group['theta']
            for p in group["params"]:
                if p.grad is None: continue
                if 'ema' not in self.state[p]:
                    self.state[p]['ema'] = p.grad.clone().detach()
                else:
                    self.state[p]['ema'].mul_(1 - theta)
                    self.state[p]['ema'].add_(p.grad, alpha=theta)


        avg_grad_norm = self._avg_grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (avg_grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = self.state[p]['ema'] * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])

        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure, which is not provided."

        self.first_step(True)
        with torch.enable_grad():
            closure()
        self.second_step()

    def _avg_grad_norm(self):
        norm = torch.norm(
            torch.stack([
                self.state[p]['ema'].norm(p=2)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm