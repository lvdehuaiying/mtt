from collections import defaultdict
from collections import Iterable
from torch.optim import Optimizer
import torch

class Multipath(Optimizer):
    def __init__(self, optimizers, k = 5, alpha = 0.5):
        self.optimizers = list(optimizers)

        self.m = len(optimizers)
        self.k = k
        self.alpha = alpha
        self.opt_p = -1

        self.param_groups = self.optimizers[0].param_groups
        for opt in optimizers:
            opt.param_groups = self.param_groups

        self.slow = dict() 
        self.state = dict()

        for group in self.param_groups:
            group['counter'] = 0

    def update_state(self, group):
         for fast in group['params']:
            if fast not in self.state:
                self.slow[fast] = torch.zeros_like(fast.data)
                self.state[fast] = torch.zeros_like(fast.data)

                self.slow[fast].copy_(fast.data)
            
            self.state[fast] += (fast.data - self.slow[fast]) * self.alpha / self.m
            fast.data.copy_(self.slow[fast])
    
    def update_slow(self, group):
        for fast in group['params']:
            self.slow[fast] += self.state[fast]
            self.state[fast] = torch.zeros_like(fast.data)

            fast.data.copy_(self.slow[fast])


    def step(self, closure=None):
        loss = self.optimizers[self.opt_p].step(closure)
        for group in self.param_groups:
            cnt = group['counter']
            if cnt % self.k == 0:
                self.update_state(group)
                self.opt_p += 1
            if cnt == 0:
                self.update_slow(group)
                self.opt_p = 0
            group['counter'] += 1
            if group['counter'] == self.m * self.k:
                group['counter'] = 0

    def state_dict(self):
        print('!!!!!!state dict!!!!!!')
