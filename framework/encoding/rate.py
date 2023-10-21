import torch
import torch.nn as nn

class poissonEncoder(nn.Module):
    def __init__(self, extend = True, T = 6) -> None:
        super(poissonEncoder,self).__init__()
        self.extend = extend
        self.T = T

    def forward(self, x):
        if(self.extend == False):
            x_spike = torch.rand_like(x).le(x).to(x)
        else:
            tx = torch.stack([torch.tensor(x) for i in range(self.T)], dim=1)
            return torch.rand_like(tx).le(tx).to(tx)
        
        return x_spike

class autoEncoder(nn.Module):
    def __init__(self, extend = True, T = 3):
        super(autoEncoder, self).__init__()
        self.extend = extend
        self.T = T

    def forward(self, x):
        if(self.extend == False):
            return x
        else:
            x_spike, _ = torch.broadcast_tensors(x, torch.zeros((self.T, ) + x.shape))
           
            return x_spike
        