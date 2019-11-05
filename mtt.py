from collections import defaultdict
from torch.optim import Optimizer
import torch

class Multipath(Optimizer):
    def __init__(self, optimizer, m = 2, k = 5, alpha = 0.5):
        self.optimizer = optimizer
        self.m = m
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
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
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            cnt = group['counter']
            if cnt % self.k == 0:
                self.update_state(group)
            if cnt == 0:
                self.update_slow(group)
            group['counter'] += 1
            if group['counter'] == self.m * self.k:
                group['counter'] = 0

    def add_param_group(self, param_group):
        param_group['counter'] = 0
        self.optimizer.add_param_group(param_group)

    def state_dict(self):
        print('!!!!!!state dict!!!!!!')
