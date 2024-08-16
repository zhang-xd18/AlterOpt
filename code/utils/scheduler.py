import math
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ['WarmUpCosineAnnealingLR', 'FakeLR', 'lamb_scheduler']


class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            return [base_lr * self.last_epoch / self.T_warmup for base_lr in self.base_lrs]
        else:
            k = 1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup))
            return [self.eta_min + (base_lr - self.eta_min) * k / 2 for base_lr in self.base_lrs]


class FakeLR(_LRScheduler):
    def __init__(self, optimizer):
        super(FakeLR, self).__init__(optimizer=optimizer)

    def get_lr(self):
        return self.base_lrs


class lamb_scheduler():
    def __init__(self, lamb_init, lamb_end, type, T):
        self.lamb_init = lamb_init
        self.lamb_end = lamb_end
        self.lamb_cur = lamb_init
        self.T = T
        self.type = type
        self.cur = 0
        
        self.a = 1/(self.T) * math.log10(lamb_end/lamb_init)
        self.b = math.log10(lamb_init)
        
        self.f = 2 * math.pi / self.T
        self.A = self.lamb_init - self.lamb_end
        self.B = (self.lamb_init + self.lamb_end)/2
        
    def init(self):
        self.cur = 0
        return 
    
    def get(self):
        return self.lamb_cur
        
    def step(self):
        if self.type == 'exp':
            self.lamb_cur = 10**(self.a * self.cur + self.b)
        elif self.type == 'cos':
            self.lamb_cur = math.cos(self.f * self.cur) * self.A / 2 + self.B
        elif self.type == 'cosh':
            self.lamb_cur = math.cos(self.f * 2 * self.cur) * self.A / 2 + self.B
        elif self.type == 'zero':
            self.lamb_cur = 0
        elif self.type == 'segment':
            if self.cur < self.T / 2:
                self.lamb_cur = 0
            else:
                self.lamb_cur = self.lamb_init
        self.cur += 1
        return 