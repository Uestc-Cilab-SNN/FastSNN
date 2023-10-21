import time

import torch
import torch.nn as nn

# from framework.neuron.cu import neuron
from framework.neuron.neuron import LIF, LifNodes
from framework.synapse.layer import Conv2d, BatchNorm2d, MaxPool2d, Linear
# from framework.datasets.vision import MNIST
from framework.datasets.MNIST import mnist

from tqdm import tqdm

from torchvision import transforms


class cnNet(nn.Module):
    def __init__(self, T, sn):
        super().__init__()
        self.T = T
        self.model = nn.Sequential(
            Conv2d(1, 32, 3, 1, 1),
            BatchNorm2d(32),
            sn,
            Conv2d(32, 32, 3, 1, 1),
            BatchNorm2d(32),
            sn,
            MaxPool2d(2, 2),
            
            Conv2d(32, 32, 3, 1, 1),
            BatchNorm2d(32),
            sn,
            Conv2d(32, 32, 3, 1, 1),
            BatchNorm2d(32),
            sn,
            MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            Linear(32 * 7 * 7, 128),
            sn,
            Linear(128, 10),
            sn,
        )
    
    def forward(self, x):
        return self.fc(
            torch.flatten(
                self.model(
                    x.unsqueeze(dim=1).repeat(1, T, 1, 1, 1)
                    ), 
                2)
            )
    

def mnist_dataset(data_dir, batch_size, test_batch_size):
    transform_train = transforms.Compose([
        # transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.85, 1.11)),
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081),
        # torch.flatten,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081),
        # torch.flatten,
    ])
    train_data_loader = torch.utils.data.DataLoader(
        dataset=mnist(
            path = data_dir,
            train = True,
            download=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=mnist(
            path = data_dir,
            train = False,
            download = True),
        batch_size = test_batch_size,
        shuffle = False,
        num_workers = 4,
        drop_last = False,
        pin_memory = True
    )
    return train_data_loader, test_data_loader


def accuracy(out, label):
    return float(out.eq(label).sum() / len(out))

def evaluate(model, dataloader, device):
    model.eval()
    pred, real = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, label = batch
            out = model(inputs.float().to(device))
            pred.extend(torch.mean(out, dim=1).argmax(dim=1))
            real.extend(label.to(device))
    model.train()
    return torch.stack(pred), torch.stack(real)


def temporal_aligned_mse(out, label):
    # out.shape [B, T, C], label.shape [B, C]
    return torch.mean(torch.square(out - label.unsqueeze(dim=1)))


if __name__ == '__main__':
    T = 6
    lr = 1e-3
    epoch = 100
    batch_size = 32
    device = torch.device(0)
    data_dir = r'playData'

    train_s, test_s = mnist_dataset(data_dir=data_dir, batch_size=batch_size, test_batch_size=batch_size)
    model = cnNet(T, LIF(sg='arctan')).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    tims = []
    accs = []

    for i in range(epoch):
        correct = 0
        total_n = 0
        tim = time.time()
        for x, y in train_s:
            out = model(x.to(device))
            optimizer.zero_grad()
            temporal_aligned_mse(out, torch.nn.functional.one_hot(y, 10).to(device)).backward()
            optimizer.step()
            correct += out.mean(dim=1).argmax(dim=1).eq(y.to(device)).sum()
            total_n += out.shape[0]

        train_acc = float(correct / total_n) * 100
        test_acc = accuracy(*evaluate(model, test_s, device)) * 100
        tim = time.time() - tim

        print('train acc {:.2f}, test acc {:.2f}, time {:.4f}'.format(train_acc, test_acc, tim))
        tims.append(tim)
        accs.append([train_acc, test_acc])

    torch.save([tims, accs], 'cupy.l')
    
    model = cnNet(T, LifNodes()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    tims = []
    accs = []

    for i in range(epoch):
        correct = 0
        total_n = 0
        tim = time.time()
        for x, y in train_s:
            out = model(x.to(device))
            optimizer.zero_grad()
            temporal_aligned_mse(out, torch.nn.functional.one_hot(y, 10).to(device)).backward()
            optimizer.step()
            correct += out.mean(dim=1).argmax(dim=1).eq(y.to(device)).sum()
            total_n += out.shape[0]

        train_acc = float(correct / total_n) * 100
        test_acc = accuracy(*evaluate(model, test_s, device)) * 100
        tim = time.time() - tim

        print('train acc {:.2f}, test acc {:.2f}, time {:.4f}'.format(train_acc, test_acc, tim))
        tims.append(tim)
        accs.append([train_acc, test_acc])

    torch.save([tims, accs], 'torch.l')