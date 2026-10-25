import torch
import numpy as np
from copy import deepcopy
import math
import torch.nn.functional as F



def bern_kl_div(p_dist, q_dist):
    p_dist_r = 1.0 - p_dist + 1.0e-8
    q_dist_r = 1.0 - q_dist + 1.0e-8

    p1 = p_dist * (p_dist.log() - q_dist.log())
    p2 = p_dist_r * (p_dist_r.log() - q_dist_r.log())

    return p1 + p2


class GammaScheduler:
    def __init__(
        self,
        initial_gamma=0.0,
        decay_steps=200,
        end_gamma=1.0,
    ):

        self.initial_gamma = initial_gamma
        self.decay_steps = decay_steps
        self.end_gamma = end_gamma

    def __call__(self, step: int) -> float:
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / self.decay_steps))
        return self.end_gamma + (self.initial_gamma - self.end_gamma) * cosine_decay

    def get_gamma(self, step: int) -> float:
        return self(step)



