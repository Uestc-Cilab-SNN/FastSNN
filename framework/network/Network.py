
from cv2 import sepFilter2D
import torch.nn as nn
import torch
import sys
from framework.neuron.neuron import LifNodes
sys.path.append("..") 
from framework.visualizing.visual import *
import framework.util as util

class Network(nn.Module):
    def __init__(self, layer_list, T, mode):
        super(Network, self).__init__()
        # names = self.__dict__
        # if mode == 'stdp':
        #     self.model = stdpNetwork(layer_list, T)

        self.layer_list = layer_list
        self.layer_num = len(layer_list)
        self.T = T
        # for i in range(self.layer_num):
        #     # names['layer' + str(i)] = layer_list[i]
        #     exec('self.layer{} = layer_list[{}]'.format(i, i))

        # self.layer0 = layer_list[0]
        # self.layer1 = layer_list[1]
        # self.layer2 = layer_list[2]
        # self.layer3 = layer_list[3]

        self.layers = nn.Sequential(*layer_list)
        self.mode = mode
        self.mem = 0
        self.spike_rate = []



    def forward(self, x):

        # if self.mode == 'stdp':
        #     return self.model(x)

        self.mem = 0
        out=[]
        y=self.layers[0](x)
        out.append(y)
        #
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        flag = False

        for i in range(1, self.layer_num):
            # if self.layer_list[i].learning_rule == 'bp':
            # if(self.layers[i].mode == 'fc' and flag == False):
                # y = y.view(y.shape[0], -1, y.shape[4])
            y=self.layers[i](y)
            out.append(y)
            # exec('out.append(self.layer{}(out[-1]))'.format(i))
            #统计最后一层模电压值用于可视化
            if i == self.layer_num - 2:
                self.mem = out[i]       #[batch, class, T]
            if(isinstance(self.layer_list[i], LifNodes)):
                rate = out[i].mean().detach().numpy()
                self.spike_rate.append(rate)
            

            # exec('out{} = self.layer{}(out{})'.format(i,i,i-1))
        # out=self.layers(x)
        if self.mode == 'rate':
            output=0
            # exec('output = torch.sum(out{}, dim=2) / 2'.format(i))
            output = torch.sum(out[-1], dim = 2) / self.T

            return output
        else:
            return out[-1]

    def visual_mem(self, batch_id):
        mem(self.mem[batch_id])

    def visual_rate(self):
        rate(self.spike_rate)


class stdpNetwork(nn.Module):
    def __init__(self, layer_list, stdp_list=None):
        super(stdpNetwork, self).__init__()

        # self.conv1 = snn.Convolution(2, 32, 5, 0.8, 0.05)
        # self.conv1_t = 10
        # self.k1 = 5
        # self.r1 = 2

        # self.conv2 = snn.Convolution(32, 150, 2, 0.8, 0.05)
        # self.conv2_t = 1
        # self.k2 = 8
        # self.r2 = 1

        # self.stdp1 = snn.STDP(self.conv1, (0.004, -0.003))
        # self.stdp2 = snn.STDP(self.conv2, (0.004, -0.003))
        self.layer_num = len(layer_list)
        self.layers = nn.Sequential(*layer_list)
        # self.stdp_layers = nn.Sequential(*stdp_list)
        self.max_ap = nn.Parameter(torch.Tensor([0.15]))

        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}
        self.spk_cnt1 = 0
        self.spk_cnt2 = 0
    
    def save_data(self, input_spike, potentials, output_spikes, winners):
        self.ctx["input_spikes"] = input_spike
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = output_spikes
        self.ctx["winners"] = winners

    def forward(self, input, max_layer):
        input = util.pad(input.float(), (2,2,2,2), 0)
        if self.training:
            # pot = self.conv1(input)
            pot = self.layers[0](input)
            spk, pot = util.fire(pot, self.layers[0].conv_t, True)
            if max_layer == 1:
                self.spk_cnt1 += 1
                if self.spk_cnt1 >= 500:
                    self.spk_cnt1 = 0
                    ap = torch.tensor(self.layers[1].learning_rate[0][0].item(), device=self.layers[1].learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.layers[1].update_all_learning_rate(ap.item(), an.item())
                pot = util.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = util.get_k_winners(pot, self.layers[0].k, self.layers[0].r, spk)
                self.save_data(input, pot, spk, winners)
                return spk, pot
            spk_in = util.pad(util.pooling(spk, 2, 2, 1), (1,1,1,1))
            spk_in = util.pointwise_inhibition(spk_in)
            pot = self.layers[2](spk_in)
            spk, pot = util.fire(pot, self.layers[2].conv_t, True)
            if max_layer == 2:
                pot = util.pointwise_inhibition(pot)
                spk = pot.sign()
                # winners = util.get_k_winners(pot, self.k2, self.r2, spk)
                winners = util.get_k_winners(pot, self.layers[2].k, self.layers[2].r, spk)
                self.save_data(spk_in, pot, spk, winners)
                return spk, pot
            spk_out = util.pooling(spk, 2, 2, 1)
            return spk_out
        else:
            pot = self.layers[0](input)
            spk, pot = util.fire(pot, self.layers[0].conv_t, True)
            pot = self.layers[2](util.pad(util.pooling(spk, 2, 2, 1), (1,1,1,1)))
            spk, pot = util.fire(pot, self.layers[2].conv_t, True)
            spk = util.pooling(spk, 2, 2, 1)
            return spk
    
    def stdp(self, layer_idx):
        if layer_idx == 1:
            self.layers[1](self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 2:
            self.layers[3](self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])



def build_net(layer_list):
    pass

