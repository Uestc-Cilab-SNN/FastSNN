import torch
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from framework.datasets.snn_dataset import SNNDataset


class mnist(SNNDataset):
    def __init__(self, path, train=True, download=True):

        super().__init__(path)
        self.path = path
        self.train = train
        self.download = download
        self.data = self.read_data()

    def read_data(self):
        if self.train:
            transform = transforms.Compose([
                transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.85, 1.11)),
                transforms.ToTensor(),
                transforms.Normalize(0.1307, 0.3081),
                #         torch.flatten
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0.1307, 0.3081),
                #         torch.flatten
            ])
        dataset = MNIST(
            root=self.path,
            train=self.train,
            transform=transform,
            download=self.download)

        return dataset

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    test_set = mnist("/")
