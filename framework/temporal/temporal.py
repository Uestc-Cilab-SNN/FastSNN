import os
import time
from typing import Union, Optional, List, Tuple
import math

import cupy as cp
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.backends import cudnn


from torch.utils.dlpack import to_dlpack as tens2dlpack
from torch.utils.cpp_extension import load as cext_load


__all__ = ['srmLinear', 'srmConv2d', 'srmNeuronFunc', 'Pooling']

_CURPATH = os.path.abspath(__file__)[:-11]

conv_wrapper = cext_load(name="conv_wrapper", sources=[os.path.join(_CURPATH, "conv_wrapper.cpp")], verbose=True)

with open(os.path.join(_CURPATH, 'C/neuron.cu'), 'r') as f:
    CU_SOURCE_CODE_RAW_STRING = f.read()


def tensor_to_cparray(ten: torch.Tensor) -> cp.ndarray:
    r""" Pack torch.Tensor to cupy.ndarray to call cupy functions.
    
    Args:
        ten (Tensor): Tensor to be packed.
    Returns:
        (cp.ndarray): An ndarray pack of input Tensor.
    
    """
    if hasattr(cp, 'core'):
        return cp.core.dlpack.fromDlpack(tens2dlpack(ten))
    else:
        return cp.from_dlpack(tens2dlpack(ten))
    

class srmNeuronFunc(object):
    r""" Implement temporal spike response model function, automatic differentiation of spike time.
    
    Args:
        funclists: cuda function names to be compiled.
        neuron_FP: srm forward function.
        neuron_BP: srm backward function.
    """
    funclists = ['srm_forward<float>', 'srm_backward<float>']
    
    cu_module = cp.RawModule(code=CU_SOURCE_CODE_RAW_STRING,
                             options=('-std=c++11', '-I ' + _CURPATH),
                             name_expressions=funclists)
    
    neuron_FP = cu_module.get_function(funclists[0])
    neuron_BP = cu_module.get_function(funclists[1])
    
    @staticmethod
    def forward(inputs: Tensor, taum: float, taus: float, e_taug: float, v_th: float) -> List[Tensor]:
        r"""
        Args:
            inputs (Tensor): pre-synapse inputs, weight * spikes.
            taum (float): Parameter in dual-exponential spike response kernel, \tau_m.
            taus (float): Parameter in dual-exponential spike response kernel, \tau_s.
            taug (float): Proxy gradient to solve reverse gradient problem.
            v_th (float): Threshold of membrane potential to emit spikes.
        Returns:
            spikes (Tensor): Spikes train, which is neuron output.
            delta_ut (Tensor): Gradient information, \partial u / \partial t_m.
            delta_u (Tensor): Gradient information, \delta t_k / \delta u.
        """
        spikes = torch.zeros_like(inputs)
        delta_ut = torch.zeros_like(inputs)
        delta_u = torch.zeros_like(inputs)
        B, T, dim = *inputs.shape[:2], inputs[0][0].numel()
        with cp.cuda.Device(inputs.get_device()):
            srmNeuronFunc.neuron_FP(((B * dim + 1023) // 1024,), (1024,), (
                tensor_to_cparray(inputs.contiguous()),
                tensor_to_cparray(spikes.contiguous()),
                tensor_to_cparray(delta_ut.contiguous()),
                tensor_to_cparray(delta_u.contiguous()),
                cp.float32(taum), cp.float32(taus), cp.float32(e_taug), cp.float32(v_th),
                cp.int32(B), cp.int32(T), cp.int32(dim)
            ))
        return spikes, delta_ut, delta_u
    
    @staticmethod
    def backward(grad_out: Tensor, delta_ut: Tensor, delta_u: Tensor, spikes: Tensor, 
                 epsw: Tensor, epst: Tensor) -> List[Tensor]:
        r"""
        Args:
            grad_out (Tensor): Gradient of the neuron output spikes from next layer.
            delta_tu (Tensor): See srmNeuronFunc.forward.
            delta_u (Tensor): See srmNeuronFunc.forward.
            spikes (Tensor): Ouput spike train.
            epsw (Tensor): Gradient information \partial u / \partial w, which is \epsion (\delta t).
            epst (Tensor): Gradient information \epsion (\delta t) / \partial t.
        Returns:
            grad_w: \partial L / \partial w.
            grad_t: \partial L / \partial t_m.
        """
        grad_w = torch.zeros_like(grad_out)
        grad_t = torch.zeros_like(grad_out)
        B, T, dim = *grad_out.shape[:2], grad_out[0][0].numel()
        with cp.cuda.Device(grad_out.get_device()):
            srmNeuronFunc.neuron_BP(((B * dim + 1023) // 1024,), (1024,), (
                tensor_to_cparray(grad_out.contiguous()), 
                tensor_to_cparray(delta_ut.contiguous()),
                tensor_to_cparray(delta_u.contiguous()), 
                tensor_to_cparray(spikes.contiguous()),
                tensor_to_cparray(epsw), 
                tensor_to_cparray(epst),
                tensor_to_cparray(grad_w.contiguous()),
                tensor_to_cparray(grad_t.contiguous()),
                cp.int32(B), cp.int32(T), cp.int32(dim)
            ))
        return grad_w, grad_t

class srmLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: Tensor, weight: Tensor, bn_weight: Optional[Tensor], bn_bias: Optional[Tensor], eps: Tensor,
                v_th: Tensor, taum: float, taus: float, e_taug: float, epsw: Tensor, epst: Tensor,
                ) -> Tensor:
        r"""
        Args:
            inputs: Input spike train from last layer.
            weight: Weight of the linear layer.
            bn_weight (Optional[Tensor]): If set to ``None``, the linear layer will not use weight normalization, 
            else the weight of the normalization.
            bn_bias (Optional[Tensor]): Same as the bn_weight.
            eps (Tensor): The value added to the denominator for numerical stability.
            ...
        Returns:
            spikes (Tensor): Output spike train.
        """
        if bn_weight is not None:
            x, normx, varx = BN1dForward(weight.t(), bn_weight, bn_bias, eps)
        else:
            x = weight.t()
            normx = varx = bn_weight = bn_bias = eps
            
        x = inputs.matmul(x)
        spikes, delta_ut, delta_u = srmNeuronFunc.forward(x, taum, taus, e_taug, v_th)
        ctx.save_for_backward(
            inputs, weight, bn_weight, bn_bias, normx, varx, eps, spikes, delta_ut, delta_u, epsw, epst, 
        )
        # print('linear', spikes.sum())
        return spikes
    
    @staticmethod
    def backward(ctx, grad_out: Tensor) -> List[Optional[Tensor]]:
        r""" Implement the calculation of the differentation of linear transformation.
        Args:
            grad_out: Gradient of the output spikes.
        Returns:
            grad_t: Gradient of input spike time.
            grad_w: Gradient of layer's weight.
            grad_bnw (Optional[Tensor]): Gradient of the weight of bn layer or None if not normalize.
            grad_bnb (Optional[Tensor]): Gradient of the bias of bn layer or None if not normalize.
        """
        inputs, weight, bn_weight, bn_bias, normx, varx, eps, spikes, delta_ut, delta_u, epsw, epst, = ctx.saved_tensors
        grad_w, grad_t = srmNeuronFunc.backward(grad_out, delta_ut, delta_u, spikes, epsw, epst)
        # grad_w: b t dout, weight: dout din, inputs: b t din
        grad_w = grad_w.transpose(1, 2).matmul(inputs).sum(dim=0)
        
        if eps.shape != bn_weight.shape or eps != bn_weight:
            grad_w, grad_bnw, grad_bnb = BN1dBackward(grad_w.t(), normx, varx, eps, bn_weight)
            grad_w = grad_w.t()
            x = (normx * bn_weight + bn_bias).t()
        else:
            grad_bnw = None
            grad_bnb = None
            x = weight
        grad_t = torch.matmul(grad_t, x)
        
        return grad_t * 0.85, grad_w, grad_bnw, grad_bnb, None, None, None, None, None, None, None

class srmLinear(nn.Linear):
    r""" Event driven based spiking fully connect layer. 
    
    Args:
        in_features (int): Input feature dimensionality.
        out_features (int): Hiddin dimensionality.
        v_th (float): Threshold of membrane potential to emit spikes. Default: 1.0.
        taum (float): Parameter in dual-exponential spike response kernel, \tau_m. Default: 5.0.
        taus (float): Parameter in dual-exponential spike response kernel, \tau_s. Default: 3.0.
        taug (float): Proxy gradient to solve reverse gradient problem, Default: 2.5.
        weight_nrom (bool): If set to ``True``, the layer will use batch norm on weight. Default: True.
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5
    """
    def __init__(self, in_features: int, out_features: int,
                 v_th: float = 1.0, taum: float = 5., taus: float = 3., taug: float = 2.5,
                 weight_norm: bool = True, eps: float = 1e-5) -> None:
        super().__init__(in_features, out_features, bias=False)
        nn.init.orthogonal_(self.weight)
        self.taum = taum
        self.taus = taus
        self.taug = taug
        self.v_th = v_th
        self.epsw = None
        self.epst = None
        self.e_taum = 1. - 1. / taum
        self.e_taus = 1. - 1. / taus
        self.e_taug = 1. - 1. / taug
        self.linear_func = srmLinearFunc.apply
        if weight_norm:
            self.bn_weight = nn.Parameter(torch.ones(out_features))
            self.bn_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bn_weight = None
            self.bn_bias = None
        self.register_buffer('eps', torch.tensor([eps]))
    
    def forward(self, inputs: Tensor) -> Tensor:
        self.batch_reset(inputs)
        return self.linear_func(
            inputs, self.weight, self.bn_weight, self.bn_bias, self.eps, self.v_th, self.taum, self.taus, self.e_taug, self.epsw, self.epst
        )
        
    def batch_reset(self, inputs: Tensor) -> None:
        r""" Reset neuron kernel if time step changed.
        
        Args:
            inputs (Tensor): Input to this layer, shape=[B, T, dim]
        """
        if self.epsw is None or self.epsw.shape[0] != inputs.shape[1]:
            coefficient = self.taum / (self.taum - self.taus)
            # for i in range(inputs.shape[1]):
            self.epst = torch.FloatTensor([-self.e_taug ** (1 + i) for i in range(inputs.shape[1])]).to(inputs)
            self.epsw = torch.FloatTensor(
                [coefficient * (self.e_taum ** (1 + i) - self.e_taus ** (1 + i)) for i in range(inputs.shape[1])]
            ).to(inputs)
    

def BN1dForward(inputs, weight, bias, eps=1e-5):
    meanx = inputs.mean(dim=0)
    varx  = inputs.var(dim=0)
    normx = (inputs - meanx) / torch.sqrt(varx + eps)
    return normx * weight + bias, normx, varx


def BN1dBackward(grad_out, normx, varx, eps, w):
    grad_bias   = grad_out.sum(dim=0)
    grad_weight = (grad_out * normx).sum(dim=0)
    grad_normx  = grad_out * w
    grad_x      = normx[:, 0].numel() * grad_normx - grad_normx.sum(dim=0) \
        - (grad_normx * normx).sum(dim=0) * normx
    grad_x      = grad_x / (normx[:, 0].numel() * torch.sqrt(varx + eps))
    return grad_x, grad_weight, grad_bias
    

class srmConvFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs: Tensor,
        weight: Tensor,
        taum: float, 
        taus: float, 
        e_taug: float, 
        v_th: float,
        epsw: Tensor,
        epst: Tensor,
        stride: Tuple[int] = (1, 1), 
        padding: Tuple[int] = (0, 0), 
        dilation: Tuple[int] = (1, 1), 
        groups: int = 1
    ) -> Tensor:
        
        out = torch.nn.functional.conv2d(
            inputs.view(-1, *inputs.shape[2:]), weight, None, stride, padding, dilation, groups
        )
        spikes, delta_ut, delta_u = srmNeuronFunc.forward(
            out.view(*inputs.shape[:2], *out.shape[1:]), taum, taus, e_taug, v_th
        )
        ctx.save_for_backward(
            inputs, weight, epsw, epst, delta_ut, delta_u, spikes, 
            torch.tensor(stride, dtype=torch.int), torch.tensor(padding, dtype=torch.int), 
            torch.tensor(dilation, dtype=torch.int), torch.tensor(groups, dtype=torch.int)
        )        
        return spikes
    
    @staticmethod
    def backward(ctx, grad_out: Tensor) -> List[Optional[Tensor]]:
        
        inputs, weight, epsw, epst, delta_ut, delta_u, spikes, stride, padding, dilation, groups = ctx.saved_tensors
        
        stride   = tuple(stride)
        padding  = tuple(padding)
        dilation = tuple(dilation)
        groups   = int(groups)
        
        grad_w, grad_t = srmNeuronFunc.backward(grad_out, delta_ut, delta_u, spikes, epsw, epst)
        
        grad_inputs = conv_wrapper.cudnn_convolution_backward_input(
            inputs.view(-1, *inputs.shape[2:]).shape, grad_t.view(-1, *grad_t.shape[2:]), weight, 
            padding, stride, dilation, groups,
            cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32
        )
        grad_inputs = grad_inputs.view(*inputs.shape) * inputs
        grad_weight = conv_wrapper.cudnn_convolution_backward_weight(
            weight.shape, grad_w.view(-1, *grad_w.shape[2:]), inputs.view(-1, *inputs.shape[2:]), 
            padding, stride, dilation, groups,
            cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32
        )
        
        return grad_inputs * 0.85, grad_weight, None, None, None, None, None, None, None, None, None, None
    
class srmConv2d(nn.Conv2d):
    r""" Event driven based spiking 2d convolutional layer.
    
    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        kernel_szie (int | tuple(int)): Size of convolving kernel.
        stride (int | tuple(int)): Stride of the convolution. Default: 1.
        padding (int | tuple | str): Padding added to both sides of the input. Default: 0.
        dilation (int | tuple(int)): Spacing between kernel elements. Default: 1.
        groups (int): Number of blocked connections from input channels to output channels. Default: 1.
        v_th (float): Threshold of membrane potential to emit spikes. Default: 1.0.
        taum (float): Parameter in dual-exponential spike response kernel, \tau_m. Default: 5.0.
        taus (float): Parameter in dual-exponential spike response kernel, \tau_s. Default: 3.0.
        taug (float): Proxy gradient to solve reverse gradient problem, Default: 2.5.
    
    """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1,
        v_th=1.0, 
        taum=5., 
        taus=3., 
        taug=2.5
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        nn.init.orthogonal_(self.weight)
        self.taum = taum
        self.taus = taus
        self.taug = taug
        self.v_th = v_th
        self.epsw = None
        self.epst = None
        self.e_taum = 1. - 1. / taum
        self.e_taus = 1. - 1. / taus
        self.e_taug = 1. - 1. / taug
        self.conv_func = srmConvFunc.apply
    
    def batch_reset(self, inputs: Tensor) -> None:
        if self.epsw is None or self.epsw.shape[0] != inputs.shape[1]:
            coefficient = self.taum / (self.taum - self.taus)
            # for i in range(inputs.shape[1]):
            self.epst = torch.FloatTensor([-self.e_taug ** (1 + i) for i in range(inputs.shape[1])]).to(inputs)
            self.epsw = torch.FloatTensor(
                [coefficient * (self.e_taum ** (1 + i) - self.e_taus ** (1 + i)) for i in range(inputs.shape[1])]
            ).to(inputs)
    
    def forward(self, inputs):
        self.batch_reset(inputs)
        return self.conv_func(
            inputs, 
            self.weight,
            self.taum, 
            self.taus, 
            self.e_taug, 
            self.v_th, 
            self.epsw, 
            self.epst, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )


class PoolFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, kernel):
        outputs = torch.nn.functional.avg_pool2d(inputs, kernel)
        ctx.save_for_backward(outputs, torch.tensor(inputs.shape), torch.tensor(kernel))
        return outputs

    @staticmethod
    def backward(ctx, grad_delta):
        (outputs, input_shape, kernel) = ctx.saved_tensors
        kernel = kernel.tolist()
        outputs = 1 / outputs
        outputs[outputs > kernel[0] * kernel[1] + 1] = 0
        outputs /= kernel[0] * kernel[1]
        grad = torch.nn.functional.interpolate(grad_delta * outputs, size=input_shape.tolist()[2:])
        return grad, None
    
class Pooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel = (2, 2)
        
    def forward(self, inputs):
        x = PoolFunc.apply(inputs.view(-1, *inputs.shape[2:]), self.kernel)
        return x.view(*inputs.shape[:2], *x.shape[1:])


class LossReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        return inputs

    @staticmethod
    def backward(ctx, grad_out):
        return -grad_out