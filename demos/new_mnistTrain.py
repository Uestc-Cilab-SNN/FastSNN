
from operator import mod
import sys
sys.path.append("..") 
from numpy import number

import torch
# from framework.datasets.vision import MNIST, NMNIST
from framework.setting import *
def setParam():
    parser.add_argument('--dt', default=.02, type=float, help='dt')
    parser.add_argument('--Tmax', default=6, type=int, help='Tmax')
    parser.add_argument('--thresh', default=1.0, type=float, help='thresh')
    parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
    parser.add_argument('--drop_rate', default=0.2, type=float, help='drop_rate')
setParam()

from framework.datasets.vision import MNIST, NMNIST
import torchvision
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn as nn
from framework.encoding.rate import poissonEncoder, autoEncoder
from framework.network.Network import Network
from framework.neuron.neuron import LifNodes
from framework.synapse.layer import Linear
from framework.visualizing.visual import showFigure 

import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_dir = './playData'
steps = 10
dt = 1

datasets1 = MNIST(
    root=dataset_dir,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True)

datasets2 = MNIST(
    root=dataset_dir,
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True)

train_dataloader = data.DataLoader(datasets1, batch_size=64)
test_loader = data.DataLoader(datasets2, batch_size=64)
encoder = poissonEncoder(extend=True, T=steps*dt)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layers = nn.Sequential(
            Linear(784, 800),
            LifNodes(),
            Linear(800, 10),
            LifNodes()
        )

    def forward(self, x):

        return torch.sum(self.layers(x), dim=1) / x.shape[1]

network = Net()

optimizer = optim.Adam(network.parameters(), lr=1e-3)
log_interval=100
for epoch in range(10):
    network.train()
    for idx, (input, label) in enumerate(train_dataloader):

        input = input.view(input.shape[0], -1)
        input = encoder(input)
        input.to(device)
        output = network(input)
        loss = F.cross_entropy(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (idx+1)%log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, idx * len(input / 2), len(train_dataloader.dataset),
                        100. * idx / len(train_dataloader), loss.item()))

    network.eval()
    test_loss = 0.0
    acc = 0.0
    with torch.no_grad():
        for idx, (input, label) in enumerate(test_loader):

            # input, label = data
            input = input.view(input.shape[0], -1)
            input = encoder(input)
            label_ = torch.zeros((label.shape[0], 10), device=label.device).scatter_(1, label.view(-1, 1), 1)
            # label_ = label.view(-1, 1)

            output = network(input)
            # test_loss += criterion(output, label_).item()
            test_loss += F.cross_entropy(output, label, reduction='sum').item()
            pred = output.argmax(dim = 1, keepdim=True)
            acc += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
        # acc /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, acc, len(test_loader.dataset),
        100. * acc / len(test_loader.dataset)))

