import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import math
import framework.encoding.temporal as temporal
import framework.encoding.rate as rate

def possion_encode(input,T,extend = True):
    encoder = rate.poissonEncoder(extend = extend,T = T)
    spikes = encoder(input)

    return spikes
def auto_encode(input,T,extend = True):
    encoder = rate.autoEncoder(extend = extend,T = T)
    spikes = encoder(input)
    l = len(list(spikes.shape))
    dims = list(range(l))
    dims[0] = 1
    dims[1] = 0
    return spikes.permute(dims)

def latency_encode(input,T):
    x = torch.abs(input)
    x_max = torch.max(x)
    x_scaled = x/x_max
    encoder = temporal.LatencyEncoder(T)
    spikes = []
    for i in range(T):
        encoded = encoder(x_scaled)
        encoded_l = encoded.tolist()
        spikes.append(encoded_l)
    spikes = torch.tensor(spikes)
    l = len(list(spikes.shape))
    dims = list(range(l))
    dims[0] = 1
    dims[1] = 0
    return spikes.permute(dims)

def Periodic_encode(input,T):

    encoder = temporal.PeriodicEncoder(input)
    spikes = []
    for i in range(T):
        encoded = encoder(input)

        encoded_l = encoded.tolist()
        spikes.append(encoded_l)
    spikes = torch.tensor(spikes)
    return spikes

def WPE_encode(input,K):
    encoder = temporal.WeightedPhaseEncoder(K)
    spikes = encoder(input)
    return spikes

def delta_encode(input,threshold=0.1,off_spike=False,):
   
    data_offset = torch.cat((torch.zeros_like(input[0]).unsqueeze(0), input))[:-1]  # add 0's to first step, remove final step

    if not off_spike:
        return torch.ones_like(input * ((input - data_offset) >= threshold))

    else:
        on_spk = torch.ones_like(input) * ((input - data_offset) >= threshold)
        off_spk = -torch.ones_like(input) * ((input - data_offset) <= -threshold)
        return on_spk + off_spk
