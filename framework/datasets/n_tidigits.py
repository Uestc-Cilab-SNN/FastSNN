import h5py

from snn_dataset import T_co
from snn_dataset import SNNDataset
from torch.utils.data import DataLoader
import numpy as np


class NTIDIGITS(SNNDataset):
    def __init__(self, path, transform=None, target_transform=None, is_train=True):
        """
        ���ݼ���ʼ��
        :param path: n-tidigits.hdf5�ļ�·��
        :param transform: ���ݴ�����
        :param target_transform: ��ǩ������
        :param is_train: ��ȡѵ������(True)�Ͷ�ȡ��������(False)
        """
        super(NTIDIGITS, self).__init__(path, transform, target_transform)
        self.is_train = is_train

        self.labels = self.get_label()
        for label in self.labels:
            print(label)

    def read_data(self, label):
        """
        ��n-tidigits.hdf5�ж�ȡ����()
        :return:
        """
        f = h5py.File(self.path, 'r')  # ��ȡ�ļ�
        if self.is_train:
            timestamps = f['train_timestamps'][label][:]  # ʱ���
            addresses = f['train_addresses'][label][:]  # ��ַ(Ƶ��ͨ��)
        else:
            timestamps = f['test_timestamps'][label][:]
            addresses = f['test_addresses'][label][:]
        timestamps = timestamps.reshape(-1, 1)
        addresses = addresses.reshape(-1, 1)
        return np.concatenate((timestamps, addresses), axis=1)  # �ϲ����¼��źŲ�����

    def get_label(self):
        """
        ��n-tidigits.hdf5�л�ȡ��ǩ
        :return:
        """
        f = h5py.File(self.path, 'r')  # ��ȡ�ļ�
        if self.is_train:  # ѵ������
            labels = f['train_labels'][:]
        else:  # ���Ա�ǩ
            labels = f['test_labels'][:]
        return labels

    def __getitem__(self, index) -> T_co:
        """
        ��������ȡ����
        :param index: ����, Ӧ��ҪС�����ݼ��ܳ���
        :return:
        """
        assert index < len(self.labels)

        data = self.read_data(self.labels[index])
        label = self.labels[index]

        if self.transform:
            data = self.transform(self.read_data(self.labels[index]))
        if self.target_transform:
            label = self.target_transform(self.labels[index])

        return data, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    file_path = './data/n-tidigits.hdf5'
    dataset = NTIDIGITS(file_path)
    dataloader = DataLoader(dataset, batch_size=1)
    for X, y in dataloader:
        print(X)
        print(y)
