import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from framework.setting import *

# batch 版本
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device =torch.device('cuda:1')

# device =torch.device('cpu')

args = parser.parse_args()

# 常调试参数
batch_size = args.batchsize  # 60
learning_rate = 1  # SGD:1 Adam: 2e-3 # 2 * 1e-4 , 1e-4
sigma = 0.05  # 0.05  # 0.1 0.05
num_epochs = 100  # max epoch
# drop_rate =0.2
drop_rate = args.drop_rate

#  不常调试参数
# thresh = 1  # neuronal threshold
thresh = args.thresh
num_classes = 10
# dt = .02
dt = args.dt
Tmax = args.Tmax
encodemax = 3
time_window = int(Tmax / dt)
encode_window = 3
nonspike = Tmax
TimeLine = torch.range(0, Tmax / dt, 1) * dt
TimeLine = TimeLine.to(device)
magnitude = 1
num_timeline = TimeLine.shape[0]
tensorZero = torch.tensor(0.).to(device)
tensorOne = torch.tensor(1.).to(device)
Timeline_reverse = (TimeLine.flip(0) + 0.1).to(device)
# lr_decay_epoch = 10

cfg_fc = [16 * 5 * 5, 1024, 128, 10]
cfg_fc_kener_size = [5, 5]
cfg_conv = [1, 6, 16]

initial_value = 0.05

"""
1.LeNet (SGD ,lr =1) : (ANN:100epoch ,batch_size = 200,SGD,test:acc0.9520)
num_epochs = 100
cfg_fc = [16*5*5, 128, 84, 10]
cfg_fc_kener_size=[5,5]
cfg_conv = [1, 6, 16]
initial_value = 0.05
train acc:99.8,test acc:98.8

"""


class Dropout_Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, drop_prob=0, training=False):

        if training:
            mask = torch.rand(size=x.shape).gt(drop_prob).float().to(device)  # 0 1矩阵
            x = x + 1  # 去掉 0
            x = x * mask
            x = x - 1
            x[x < 0] = Tmax
            out = x
            ctx.save_for_backward(mask, )

            return out
        else:
            mask = torch.ones_like(x).float().to(device)
            ctx.save_for_backward(mask, )
            return x

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors

        return grad_output * mask, None, None


dropout = Dropout_Spike.apply


# class Dropout_Spike(nn.Module):
#     def __init__(self, p):
#         super(Dropout_Spike, self).__init__()
#         self.prob = p
#
#     def forward(self, x, training):
#         if training:
#
#             mask = torch.rand(size=x.shape).gt(self.prob).float().to(device)  # 0 1矩阵
#             x += 1  #去掉 0
#             x *= mask
#             x -= 1
#             x[x <0] = Tmax
#             return x
#         else:
#             return x

def mask_kernel(channel, height, width):
    index_kernel = torch.arange(channel * height * width * 3).reshape(channel * height * width, 3)

    index_kernel[:, 0] = index_kernel[:, 0] / (height * width * 3) % channel
    index_kernel[:, 1] = index_kernel[:, 1] / (width * 3) % height
    index_kernel[:, 2] = index_kernel[:, 2] / 3 % width

    full_indices = torch.cat((index_kernel, index_kernel), 1).t().numpy().tolist()
    kernel_mask = torch.zeros((channel, height, width, channel, height, width))
    kernel_mask[full_indices] = 1

    return kernel_mask.reshape((channel * height * width, channel, height, width)).to(device)


def voltage_relay(voltage_con, weight):
    voltage_transpose = voltage_con.permute(0, 2, 1)
    weight_transpose = weight.t()
    voltage_post = voltage_transpose.matmul(weight_transpose)
    voltage_post = voltage_post.permute(0, 2, 1)
    return voltage_post


def seek_spike(voltage_post):
    voltage_binary = torch.where(voltage_post >= thresh, tensorOne, tensorZero)
    voltage_binary[:, :, -1] = 1.
    voltage_binary *= Timeline_reverse
    spike_time = voltage_binary.argmax(2).type(torch.float)
    return spike_time.to(device) * dt


def seek_spike_conv(mem):
    voltage_binary = torch.where(mem >= thresh, tensorOne, tensorZero)
    voltage_binary[:, :, :, :, -1] = 1.
    voltage_binary *= Timeline_reverse
    spike_time = voltage_binary.argmax(4).type(torch.float)

    return spike_time.to(device) * dt


class Linear_Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ti, W):  # dropout 用于全连接层
        # 计算膜电压
        W_shape = W.shape
        shape = ti.shape
        ti = ti.view(batch_size, -1)
        subtract_input = torch.repeat_interleave(ti.reshape(shape[0], shape[1], 1), num_timeline, dim=2)
        tmp = F.relu(TimeLine - subtract_input)
        mem = voltage_relay(tmp, W)
        # 计算脉冲to
        out = seek_spike(mem)
        ctx.save_for_backward(torch.autograd.Variable(out), torch.autograd.Variable(ti), torch.autograd.Variable(W), )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # 注意该处的grad_output 为dE/dto
        out, ti, W, = ctx.saved_tensors
        dEdto = grad_output
        full_out = torch.repeat_interleave(out.reshape(out.shape[0], out.shape[1], 1), ti.shape[1], dim=2)
        full_ti = torch.repeat_interleave(ti.reshape(ti.shape[0], 1, ti.shape[1]), out.shape[1], dim=1)

        dvdw = F.relu(full_out - full_ti)
        mask = dvdw.gt(0).float()

        # #该函数中需要反传的是dt/dv
        dtdv = -1 / (torch.mul(mask, W).sum(dim=2))
        dtdv = torch.where(out == nonspike, tensorZero, dtdv)
        dtdv = dtdv.clamp(min=-1)

        # 这一步需要反向计算的梯度是 dv/dti
        # b*o*i
        dvdti = mask * (-torch.repeat_interleave(W.reshape(1, W.shape[0], W.shape[1]), batch_size, dim=0))

        dvdti = torch.where(torch.isnan(dvdti) == 1, torch.zeros_like(dvdti), dvdti)

        if dEdto.shape[1] == cfg_fc[-1]:
            ind_label = dEdto > 0
            ind_nonspike = out == nonspike
            ind_target = ind_label.float() * ind_nonspike.float()
            dtdv = torch.where(ind_target == 1, -tensorOne, dtdv)

        dvdw = torch.where(dvdw == nonspike, torch.zeros_like(dvdw), dvdw)

        dvdw = dvdw.clamp(max=2)
        dvdw = torch.where(torch.isnan(dvdw) == 1, torch.zeros_like(dvdw), dvdw)

        delta = dEdto * dtdv
        dE = delta.reshape(delta.shape[0], 1, delta.shape[1]).matmul(dvdti)
        dE = dE.squeeze(dim=1)

        deriv = torch.mul(
            torch.repeat_interleave(delta.reshape(delta.shape[0], delta.shape[1], 1), dvdw.shape[2], dim=2), dvdw)
        deriv = torch.sum(deriv, dim=0)
        deriv /= batch_size

        return dE, deriv


class Conv2d_Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ti, W):  # dropout 用于全连接层
        # 计算膜电压
        shape = ti.shape
        tmp = F.relu(TimeLine - ti.reshape(shape[0], shape[1], shape[2], shape[3], 1))

        # size in b*time*channels*width*height
        tmp = tmp.permute(0, 4, 1, 2, 3)
        ts = tmp.shape
        tmp = tmp.reshape(ts[0] * ts[1], ts[2], ts[3], ts[4])
        mem = F.conv2d(tmp, W)
        mem = mem.reshape(batch_size, num_timeline, mem.shape[1], mem.shape[2], mem.shape[3]).permute(0, 2, 3, 4, 1)
        # out_shape = mem.shape

        out = seek_spike_conv(mem)

        ctx.save_for_backward(torch.autograd.Variable(out), torch.autograd.Variable(ti), torch.autograd.Variable(W))

        return out

    @staticmethod
    def backward(ctx, grad_output):
        # 注意该处的grad_output 为dE/dto
        out, ti, W = ctx.saved_tensors
        # 求出dt/dv、dv/dw、dv/dti
        W_shape = W.shape
        shape = ti.shape
        conv_window = W_shape[2]
        out_channel = W_shape[0]
        in_channel = W_shape[1]
        out_shape = out.shape
        # 求出dv/dw
        input_full = F.conv2d(ti, mask_kernel(in_channel, conv_window, conv_window)) \
            .reshape(shape[0], 1, shape[1], conv_window, conv_window, out_shape[2], out_shape[3])

        # size in b*o*i*5*5*h'*w'

        dvdw = F.relu(out.reshape(out_shape[0], out_shape[1], 1, 1, 1, out_shape[2], out_shape[3]) - input_full)
        # size in b*o*i*h'*w'*5*5
        mask = dvdw.permute(0, 1, 2, 5, 6, 3, 4).gt(0).float()
        # 求出dt/dv

        dtdv = torch.mul(mask, W.reshape(1, W.shape[0], W.shape[1], 1, 1, conv_window, conv_window))

        dtdv = dtdv.sum(dim=6).sum(dim=5).sum(dim=2)
        # size in b*o*h'*w'
        dtdv = -1 / dtdv
        dtdv = torch.where(out == nonspike, tensorZero, dtdv)
        dtdv = dtdv.clamp(min=-1)

        # 求出dv/dti
        out_padding = torch.nn.functional.pad(out, (conv_window - 1, conv_window - 1, conv_window - 1, conv_window - 1),
                                              'constant', 0)
        mask_out2in = F.conv2d(out_padding, mask_kernel(out_channel, conv_window, conv_window)).reshape(
            batch_size, out_shape[1], 1, conv_window,
            conv_window, shape[2], shape[3])
        mask_out2in = (mask_out2in - ti.reshape(batch_size, 1, shape[1], 1, 1, shape[2], shape[3])).gt(0).float()
        dvdti = mask_out2in.permute(0, 1, 2, 5, 6, 3, 4) * (
            torch.flip(-W, [2, 3]).reshape(1, W_shape[0], W_shape[1], 1, 1, conv_window, conv_window))

        dvdti = torch.where(torch.isnan(dvdti) == 1, tensorZero, dvdti)

        conv_window = W_shape[2]
        out_channel = W_shape[0]
        in_channel = W_shape[1]
        # b*o*h'*w'

        dEdto = grad_output

        dvdw = torch.where(dvdw == nonspike, tensorZero, dvdw)

        dvdw = dvdw.clamp(max=2)
        dvdw = torch.where(torch.isnan(dvdw) == 1, tensorZero, dvdw)

        # why does this variable contain nan?
        # size in b*o*h'*w'
        delta = dEdto * dtdv

        dvdw_shape = dvdw.shape
        deriv = torch.mul(delta.reshape(batch_size, dvdw_shape[1], 1, 1, 1, dvdw_shape[5], dvdw_shape[6]), dvdw)
        deriv = deriv.sum(dim=6).sum(dim=5).sum(dim=0)

        delta_padding = torch.nn.functional.pad(delta, (conv_window - 1, conv_window - 1, conv_window - 1,
                                                        conv_window - 1), 'constant', 0)
        vti_shape = dvdti.shape

        delta_padding = F.conv2d(delta_padding, mask_kernel(out_channel, conv_window, conv_window)). \
            reshape(vti_shape[0], vti_shape[1], 1, conv_window, conv_window, vti_shape[3], vti_shape[4])
        dE = delta_padding.permute(0, 1, 2, 5, 6, 3, 4) * dvdti
        dE = dE.sum(dim=6).sum(dim=5).sum(dim=1)

        deriv *= (magnitude / batch_size)

        return dE, deriv


Linear = Linear_Spike.apply
Conv2d = Conv2d_Spike.apply


class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()

        self.conv1_weight = nn.Parameter(
            torch.randn(cfg_conv[1], cfg_conv[0], cfg_fc_kener_size[0], cfg_fc_kener_size[0])
            * initial_value + initial_value, requires_grad=True)

        self.conv2_weight = nn.Parameter(
            torch.randn(cfg_conv[2], cfg_conv[1], cfg_fc_kener_size[1], cfg_fc_kener_size[1])
            * 0.1 * initial_value + 0.1 * initial_value, requires_grad=True)

        # self.conv3_weight = nn.Parameter(torch.randn(cfg_conv[3], cfg_conv[2], cfg_fc_kener_size[2],cfg_fc_kener_size[2])
        #                                  * 0.1 * initial_value + 0.1 * initial_value,requires_grad=True)

        self.fc1_weight = nn.Parameter(torch.randn(cfg_fc[1], cfg_fc[0]) * 0.01 + 0.01, requires_grad=True)
        self.fc2_weight = nn.Parameter(torch.randn(cfg_fc[2], cfg_fc[1]) * 0.01 + 0.01, requires_grad=True)
        self.fc3_weight = nn.Parameter(torch.randn(cfg_fc[3], cfg_fc[2]) * 0.01 + 0.01, requires_grad=True)
        self.conv_spike = Conv2d
        self.linear_spike = Linear
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, input):
        ti = input
        c1_spike = self.conv_spike(ti, self.conv1_weight)
        p1_spike = -self.pool(-c1_spike)
        c2_spike = self.conv_spike(p1_spike, self.conv2_weight)
        p2_spike = -self.pool(-c2_spike)
        # c3_spike = mem_update(p2_spike, self.conv3_weight,self.conv )
        # f0_spike = c3_spike.reshape(batch_size, cfg_fc[0])
        f0_spike = p2_spike.reshape(batch_size, cfg_fc[0])
        h1_spike = self.linear_spike(f0_spike, self.fc1_weight)
        h1_spike = dropout(h1_spike, drop_rate, self.training)
        h2_spike = self.linear_spike(h1_spike, self.fc2_weight)
        h2_spike = dropout(h2_spike, drop_rate, self.training)
        h3_spike = self.linear_spike(h2_spike, self.fc3_weight)
        # outputs = h2_spike
        outputs = h3_spike
        return outputs


class loss_fun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, output, label):
        # 计算出loss
        to = output.detach()
        mask = label.detach()
        a = F.softmax(-to, dim=1)

        loss_distri = -torch.log((mask * a).sum(dim=1))

        loss = loss_distri.sum() / batch_size

        is_update = (loss_distri < 0.5)

        # 反传dE/dto
        a = torch.where(to == nonspike, 0 * torch.ones_like(to), a)
        ctx.save_for_backward(a, mask, is_update)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        a, mask, is_update = ctx.saved_tensors
        dEdt = torch.where(mask == 1, 1 - a, -a)
        ud_mask = torch.repeat_interleave(is_update.reshape(is_update.shape[0], 1), dEdt.shape[1], dim=1)
        dEdt[ud_mask] = 0
        return dEdt, None,


#------------------------stdp utils----------------------------
# padding
# pad = (padLeft, padRight, padTop, padBottom)
def pad(input, pad, value=0):
    r"""Applies 2D padding on the input tensor.

    Args:
        input (Tensor): The input tensor.
        pad (tuple): A tuple of 4 integers in the form of (padLeft, padRight, padTop, padBottom)
        value (int or float): The value of padding. Default: 0

    Returns:
        Tensor: Padded tensor.
    """
    return F.pad(input, pad, value=value)

# pooling
def pooling(input, kernel_size, stride=None, padding=0):
    r"""Performs a 2D max-pooling over an input signal (spike-wave or potentials) composed of several input
    planes.

    Args:
        input (Tensor): The input tensor.
        kernel_size (int or tuple): Size of the pooling window.
        stride (int or tuple, optional): Stride of the pooling window. Default: None
        padding (int or tuple, optional): Size of the padding. Default: 0

    Returns:
        Tensor: The result of the max-pooling operation.
    """
    return F.max_pool2d(input, kernel_size, stride, padding)

def fire(potentials, threshold=None, return_thresholded_potentials=False):
    r"""Computes the spike-wave tensor from tensor of potentials. If :attr:`threshold` is :attr:`None`, all the neurons
    emit one spike (if the potential is greater than zero) in the last time step.

    Args:
        potentials (Tensor): The tensor of input potentials.
        threshold (float): Firing threshold. Default: None
        return_thresholded_potentials (boolean): If True, the tensor of thresholded potentials will be returned
        as well as the tensor of spike-wave. Default: False

    Returns:
        Tensor: Spike-wave tensor.
    """
    thresholded = potentials.clone().detach()
    if threshold is None:
        thresholded[:-1]=0
    else:
        F.threshold_(thresholded, threshold, 0)
    if return_thresholded_potentials:
        return thresholded.sign(), thresholded
    return thresholded.sign()

def fire_(potentials, threshold=None):
    r"""The inplace version of :func:`~fire`
    """
    if threshold is None:
        potentials[:-1]=0
    else:
        F.threshold_(potentials, threshold, 0)
    potentials.sign_()

def threshold(potentials, threshold=None):
    r"""Applies a threshold on potentials by which all of the values lower or equal to the threshold becomes zero.
    If :attr:`threshold` is :attr:`None`, only the potentials corresponding to the final time step will survive.

    Args:
        potentials (Tensor): The tensor of input potentials.
        threshold (float): The threshold value. Default: None

    Returns:
        Tensor: Thresholded potentials.
    """
    outputs = potentials.clone().detach()
    if threshold is None:
        outputs[:-1]=0
    else:
        F.threshold_(outputs, threshold, 0)
    return outputs

def threshold_(potentials, threshold=None):
    r"""The inplace version of :func:`~threshold`
    """
    if threshold is None:
        potentials[:-1]=0
    else:
        F.threshold_(potentials, threshold, 0)

# in each position, the most fitted feature will survive (first earliest spike then maximum potential)
# it is assumed that the threshold function is applied on the input potentials
def pointwise_inhibition(thresholded_potentials):
    r"""Performs point-wise inhibition between feature maps. After inhibition, at most one neuron is allowed to fire at each
    position, which is the neuron with the earliest spike time. If the spike times are the same, the neuron with the maximum
    potential will be chosen. As a result, the potential of all of the inhibited neurons will be reset to zero.

    Args:
        thresholded_potentials (Tensor): The tensor of thresholded input potentials.

    Returns:
        Tensor: Inhibited potentials.
    """
    # maximum of each position in each time step
    maximum = torch.max(thresholded_potentials, dim=1, keepdim=True)
    # compute signs for detection of the earliest spike
    clamp_pot = maximum[0].sign()
    # maximum of clamped values is the indices of the earliest spikes
    clamp_pot_max_1 = (clamp_pot.size(0) - clamp_pot.sum(dim = 0, keepdim=True)).long()
    clamp_pot_max_1.clamp_(0,clamp_pot.size(0)-1)
    clamp_pot_max_0 = clamp_pot[-1:,:,:,:]
    # finding winners (maximum potentials between early spikes)
    winners = maximum[1].gather(0, clamp_pot_max_1)
    # generating inhibition coefficient
    coef = torch.zeros_like(thresholded_potentials[0]).unsqueeze_(0)
    coef.scatter_(1, winners,clamp_pot_max_0)
    # applying inhibition to potentials (broadcasting multiplication)
    return torch.mul(thresholded_potentials, coef)

# inhibiting particular features, preventing them to be winners
# inhibited_features is a list of features numbers to be inhibited
def feature_inhibition_(potentials, inhibited_features):
    r"""The inplace version of :func:`~feature_inhibition`
    """
    if len(inhibited_features) != 0:
        potentials[:, inhibited_features, :, :] = 0

def feature_inhibition(potentials, inhibited_features):
    r"""Inhibits specified features (reset the corresponding neurons' potentials to zero).

    Args:
        potentials (Tensor): The tensor of input potentials.
        inhibited_features (List): The list of features to be inhibited.

    Returns:
        Tensor: Inhibited potentials.
    """
    potentials_copy = potentials.clone().detach()
    if len(inhibited_features) != 0:
        feature_inhibition_(potentials_copy, inhibited_features)
    return potentials_copy

# returns list of winners
# inhibition_radius is to increase the chance of diversity among features (if needed)
def get_k_winners(potentials, kwta = 1, inhibition_radius = 0, spikes = None):
    r"""Finds at most :attr:`kwta` winners first based on the earliest spike time, then based on the maximum potential.
    It returns a list of winners, each in a tuple of form (feature, row, column).

    .. note::

        Winners are selected sequentially. Each winner inhibits surrounding neruons in a specific radius in all of the
        other feature maps. Note that only one winner can be selected from each feature map.

    Args:
        potentials (Tensor): The tensor of input potentials.
        kwta (int, optional): The number of winners. Default: 1
        inhibition_radius (int, optional): The radius of lateral inhibition. Default: 0
        spikes (Tensor, optional): Spike-wave corresponding to the input potentials. Default: None

    Returns:
        List: List of winners.
    """
    if spikes is None:
        spikes = potentials.sign()
    # finding earliest potentials for each position in each feature
    maximum = (spikes.size(0) - spikes.sum(dim = 0, keepdim=True)).long()
    maximum.clamp_(0,spikes.size(0)-1)
    values = potentials.gather(dim=0, index=maximum) # gathering values
    # propagating the earliest potential through the whole timesteps
    truncated_pot = spikes * values
    # summation with a high enough value (maximum of potential summation over timesteps) at spike positions
    v = truncated_pot.max() * potentials.size(0)
    truncated_pot.addcmul_(spikes,v)
    # summation over all timesteps
    total = truncated_pot.sum(dim=0,keepdim=True)
    
    total.squeeze_(0)
    global_pooling_size = tuple(total.size())
    winners = []
    for k in range(kwta):
        max_val,max_idx = total.view(-1).max(0)
        if max_val.item() != 0:
            # finding the 3d position of the maximum value
            max_idx_unraveled = np.unravel_index(max_idx.item(),global_pooling_size)
            # adding to the winners list
            winners.append(max_idx_unraveled)
            # preventing the same feature to be the next winner
            total[max_idx_unraveled[0],:,:] = 0
            # columnar inhibition (increasing the chance of leanring diverse features)
            if inhibition_radius != 0:
                rowMin,rowMax = max(0,max_idx_unraveled[-2]-inhibition_radius),min(total.size(-2),max_idx_unraveled[-2]+inhibition_radius+1)
                colMin,colMax = max(0,max_idx_unraveled[-1]-inhibition_radius),min(total.size(-1),max_idx_unraveled[-1]+inhibition_radius+1)
                total[:,rowMin:rowMax,colMin:colMax] = 0
        else:
            break
    return winners

# decrease lateral intencities by factors given in the inhibition_kernel
def intensity_lateral_inhibition(intencities, inhibition_kernel):
    r"""Applies lateral inhibition on intensities. For each location, this inhibition decreases the intensity of the
    surrounding cells that has lower intensities by a specific factor. This factor is relative to the distance of the
    neighbors and are put in the :attr:`inhibition_kernel`.

    Args:
        intencities (Tensor): The tensor of input intensities.
        inhibition_kernel (Tensor): The tensor of inhibition factors.

    Returns:
        Tensor: Inhibited intensities.
    """
    intencities.squeeze_(0)
    intencities.unsqueeze_(1)

    inh_win_size = inhibition_kernel.size(-1)
    rad = inh_win_size//2
    # repeat each value
    values = intencities.reshape(intencities.size(0),intencities.size(1),-1,1)
    values = values.repeat(1,1,1,inh_win_size)
    values = values.reshape(intencities.size(0),intencities.size(1),-1,intencities.size(-1)*inh_win_size)
    values = values.repeat(1,1,1,inh_win_size)
    values = values.reshape(intencities.size(0),intencities.size(1),-1,intencities.size(-1)*inh_win_size)
    # extend patches
    padded = F.pad(intencities,(rad,rad,rad,rad))
    # column-wise
    patches = padded.unfold(-1,inh_win_size,1)
    patches = patches.reshape(patches.size(0),patches.size(1),patches.size(2),-1,patches.size(3)*patches.size(4))
    patches.squeeze_(-2)
    # row-wise
    patches = patches.unfold(-2,inh_win_size,1).transpose(-1,-2)
    patches = patches.reshape(patches.size(0),patches.size(1),1,-1,patches.size(-1))
    patches.squeeze_(-3)
    # compare each element by its neighbors
    coef = values - patches
    coef.clamp_(min=0).sign_() # "ones" are neighbors greater than center
    # convolution with full stride to get accumulative inhibiiton factor
    factors = F.conv2d(coef, inhibition_kernel, stride=inh_win_size)
    result = intencities + intencities * factors

    intencities.squeeze_(1)
    intencities.unsqueeze_(0)
    result.squeeze_(1)
    result.unsqueeze_(0)
    return result

# performs local normalization
# on each region (of size radius*2 + 1) the mean value is computed and 
# intensities will be divided by the mean value
# x is a 4D tensor
def local_normalization(input, normalization_radius, eps=1e-12):
    r"""Applies local normalization. on each region (of size radius*2 + 1) the mean value is computed and the
    intensities will be divided by the mean value. The input is a 4D tensor.

    Args:
        input (Tensor): The input tensor of shape (timesteps, features, height, width).
        normalization_radius (int): The radius of normalization window.

    Returns:
        Tensor: Locally normalized tensor.
    """
    # computing local mean by 2d convolution
    kernel = torch.ones(1,1,normalization_radius*2+1,normalization_radius*2+1,device=input.device).float()/((normalization_radius*2+1)**2)
    # rearrange 4D tensor so input channels will be considered as minibatches
    y = input.squeeze(0) # removes minibatch dim which was 1
    y.unsqueeze_(1)  # adds a dimension after channels so previous channels are now minibatches
    means = F.conv2d(y,kernel,padding=normalization_radius) + eps # computes means
    y = y/means # normalization
    # swap minibatch with channels
    y.squeeze_(1)
    y.unsqueeze_(0)
    return y





