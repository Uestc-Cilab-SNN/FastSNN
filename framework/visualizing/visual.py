import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def mem(mem): #[batch,class,T]
    # print(mem.shape)
    cls_cnt = mem.shape[0]

    mem = mem[..., -1]
    # print(mem.shape)
    mem = mem.detach().numpy()  #[class]

    x_dim=np.arange(1,cls_cnt+1)
    plt.bar(x_dim,mem)
    plt.show()

def rate(rate):
    print(len(rate), rate)
    ax=plt.gca()
    x_dim=np.arange(1,len(rate)+1)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.scatter(x_dim,rate)
    plt.show()

def showFigure(figs,i):
    fig = figs[i]
    fig = torch.sum(fig, dim=-1) / fig.shape[-1]
    fig *= 255
    fig = fig.permute(1,2,0)
    plt.imshow(fig)
    plt.axis('off')
    plt.show()