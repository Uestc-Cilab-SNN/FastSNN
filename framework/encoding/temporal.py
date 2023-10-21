import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import math
from framework.encoding.base import StatefulEncoder

class LatencyEncoder(StatefulEncoder):
    def __init__(self, T: int, enc_function='linear'):
    
        super().__init__(T)
        if enc_function == 'log':
            self.alpha = math.exp(T - 1.) - 1.
        elif enc_function != 'linear':
            raise NotImplementedError

        self.enc_function = enc_function

    def encode(self,x: torch.Tensor):
        if self.enc_function == 'log':
            t_f = (self.T - 1. - torch.log(self.alpha * x + 1.)).round().long()
        else:
            t_f = ((self.T - 1.) * (1. - x)).round().long()

        self.spike = F.one_hot(t_f, num_classes=self.T).to(x)
        # [*, T] -> [T, *]
        d_seq = list(range(self.spike.ndim - 1))
        d_seq.insert(0, self.spike.ndim - 1)
        self.spike = self.spike.permute(d_seq)

class PeriodicEncoder(StatefulEncoder):
    def __init__(self, spike: torch.Tensor):
        super().__init__(spike.shape[0])
        self.encode(spike)

    def encode(self, spike: torch.Tensor):
        self.spike = spike
        self.T = spike.shape[0]

class WeightedPhaseEncoder(StatefulEncoder):
    def __init__(self, K: int):
     
        super().__init__(K)

    def encode(self, x: torch.Tensor):
        if (x.all() >=0) and (x.all() <= 1 - 2 ** (-self.T)):
            x_clone = x.clone()
            self.spike = torch.empty((self.T,) + x.shape, device=x.device)  # Encoding to [T, batch_size, *]
            w = 0.5
            for i in range(self.T):
                self.spike[i] = x_clone >= w
                x_clone -= w * self.spike[i]
                w *= 0.5
            return self.spike
        else:
            raise ValueError

