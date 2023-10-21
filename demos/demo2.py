import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from framework.temporal.temporal import srmLinear, srmConv2d, Pooling, LossReverse
from framework.synapse.layer import MaxPool2d, AvgPool2d
from framework.encoding.rate import poissonEncoder, autoEncoder
# from framework.datasets.vision import MNIST
from framework.datasets.MNIST import mnist


class ConvNet(nn.Module):
    def __init__(self, T):
        self.T = T
        super().__init__()
        self.model = nn.Sequential(
            srmConv2d(1, 15, 5),
            Pooling(),
            srmConv2d(15, 40, 5),
            Pooling()
        )
        self.out = nn.Sequential(
            srmLinear(640, 300),
            srmLinear(300, 10)
            # srmLinear(32 * 7 * 7, 10)
        )
        
    def forward(self, inputs):
        h = self.model(inputs.unsqueeze(dim=1).repeat(1, self.T, 1, 1, 1))
        out = self.out(torch.flatten(h, 2))
        return out


def mnist_dataset(data_dir, batch_size, test_batch_size):

    train_data_loader = torch.utils.data.DataLoader(
        dataset=mnist(
            path=data_dir,
            train=True,
            download=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset=mnist(
            path=data_dir,
            train=False,
            download=True),
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True
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
    lr = 5e-4
    epoch = 200
    batch_size = 128
    # device = torch.device(0)
    device = torch.device('cuda')
    data_dir = '.'
    train_s, test_s = mnist_dataset('playData', batch_size, 1024)
    model = ConvNet(T).to(device)
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch, eta_min=0)
    encoder = poissonEncoder(T=T)

    tims = []
    accs = []
    
    for i in range(epoch):
        correct = 0
        total_n = 0
        tim = time.time()
        model.train()
        for j, (x, y) in enumerate(tqdm(train_s)):
            out = model(x.to(device))
            if j == 0: print(out.sum())
            optimizer.zero_grad()
            loss = torch.square(LossReverse.apply(out).mean(dim=1) - torch.nn.functional.one_hot(y.to(device), 10)).sum()
            loss.backward()
            optimizer.step()
            correct += out.mean(dim=1).argmax(dim=1).eq(y.to(device)).sum()
            total_n += out.shape[0]
            # print(correct / total_n)
            if (1 + j) % 100 == 0: 
                print(out.sum())
                print(loss)
                print(correct / total_n)
        
        train_acc = float(correct / total_n) * 100
        test_acc = accuracy(*evaluate(model, test_s, device)) * 100
        tim = time.time() - tim
        
        print('train acc {:.2f}, test acc {:.2f}, time {:.4f}'.format(train_acc, test_acc, tim))
        tims.append(tim)
        accs.append([train_acc, test_acc])
    
    torch.save([tims, accs], 'cupy.l')