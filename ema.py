import torch

class ExponentialMovingAverage():
    def __init__(self, optim, decay):
        self.optim = optim
        self.state = {}
        self.backup = {}
        self.decay = decay

    def update(self):
        for group in self.optim.param_groups:
            for p in group['params']:
                d_p = p.data
                if p not in self.state:
                    self.state[p] = torch.clone(d_p).detach()
                    continue
                if p.grad is None:
                    continue

                self.state[p].mul_(self.decay).add_(1-self.decay, d_p)

    def apply_state(self):
        for group in self.optim.param_groups:
            for p in group['params']:
                assert(p in self.state)

                self.backup[p] = p.data
                p.data = self.state[p]

    def restore(self):
        for group in self.optim.param_groups:
            for p in group['params']:
                assert(p in self.backup)

                p.data = self.backup[p]
                self.backup = {}
