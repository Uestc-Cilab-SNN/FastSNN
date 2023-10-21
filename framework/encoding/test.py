
import framework.encoding.encode as encode
import framework.encoding.temporal as temporal
import framework.encoding.rate as rate
import torch.utils.data as data
import torch
from framework.datasets.vision import MNIST, NMNIST
import torchvision


dataset_dir = './playData'

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

train_dataloader = data.DataLoader(datasets1, batch_size=1)
test_loader = data.DataLoader(datasets2, batch_size=64)
T = 5

for x,y in train_dataloader:
    print("x_shape",x.shape)
    possoin = encode.possion_encode(x,T,extend = True)
    print("possoin",possoin.shape)

    auto = encode.auto_encode(x,T)
    print("auto",auto.shape)

    latency = encode.latency_encode(x,T)
    print("latency",latency.shape)

    delta = encode.delta_encode(x,0.1)
    print("delta",delta.shape)

    WPE = encode.WPE_encode(x,T)
    print("WPE",WPE.shape)  

    Periodic = encode.Periodic_encode(x,T)
    print("Periodic",Periodic.shape)

    break
