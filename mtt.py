from collections import defaultdict
from collections import Iterable
from torch.optim import Optimizer
import torch

import math

class Multipath(Optimizer):
    def __init__(self, optimizers, v_s = None, k = 5, alpha = 0.5):
        assert(math.isclose(sum(v_s), 1))
        assert(max(v_s) <= 1)
        assert(min(v_s) > 0)

        self.optimizers = list(optimizers)

        self.m = len(optimizers)
        self.k = k
        self.alpha = alpha
        self.opt_p = 0

        self.param_groups = self.optimizers[0].param_groups
        for opt in optimizers:
            opt.param_groups = self.param_groups

        self.slow = dict() 
        self.state = dict()

        for group in self.param_groups:
            group['counter'] = 0

            for fast in group['params']:
                self.state[fast] = torch.zeros_like(fast)
                self.slow[fast] = torch.clone(fast).detach()

        self.v_s = v_s
        if v_s is None:
            self.v_s = [1 / self.m for _ in range(self.m)]

    def update_state(self, group, v):
        for fast in group['params']:
            assert(fast in self.state)
            
            self.state[fast] += (fast.data - self.slow[fast]) * self.alpha * v
            fast.data.copy_(self.slow[fast])
    
    def update_slow(self, group):
        for fast in group['params']:
            self.slow[fast] += self.state[fast]
            self.state[fast] = torch.zeros_like(fast.data)

            fast.data.copy_(self.slow[fast])


    def step(self, closure=None):
        loss = self.optimizers[self.opt_p].step(closure)
        for group in self.param_groups:
            group['counter'] += 1
            cnt = group['counter']
            if cnt % self.k == 0:
                self.update_state(group, self.v_s[self.opt_p])
                self.opt_p += 1
            if group['counter'] == self.m * self.k:
                self.update_slow(group)
                self.opt_p = 0
                group['counter'] = 0


