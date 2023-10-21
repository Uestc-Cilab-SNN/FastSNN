import os

from snn_dataset import T_co
from snn_dataset import SNNDataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from PIL import Image


class NYU2(SNNDataset):
    def __init__(self, path, transform=None, target_transform=None):
        """
        ���ݼ���ʼ��
        :param path: ���ݼ���·��
        :param transform: �������ݴ�����
        :param target_transform: ��ǩ���ݴ�����
        """
        super(NYU2, self).__init__(path, transform, target_transform)

        self.data = self.read_data()

    def read_data(self):
        """
        ��ȡ����
        :return:
        """
        data_dir = self.path + '/data'  # �����ļ���
        paths = os.listdir(data_dir)  # ��ȡ�ļ��������е��ļ���
        for i in range(len(paths)):
            paths[i] = data_dir + '/' + paths[i]  # �ļ���ת��Ϊ�ļ�·��
        return paths

    def get_label(self, data_path):
        """
        �ڱ����ݼ���, ���Դ����ݵ��ļ������ҵ���ǩ���ļ���, �Ӷ���ȡ��ǩͼƬ��·��
        :param data_path: ���ݵ��ļ�·��, ����: ./data/ȥ�����ݼ�/data/NYU2_523_2_3.jpg
        :return:
        """
        data_name = data_path.split('/')[-1][0:-8]  # ��ȡ���ݵ��ļ���
        label_path = self.path + '/label/' + data_name + '.jpg'
        return label_path

    def __getitem__(self, index) -> T_co:
        """
        ��ȡ����, ��ǩ
        :param index: ����
        :return:
        """
        data_path = self.data[index]  # ����·��
        label_path = self.get_label(data_path)  # ��ǩ·��

        data_image = Image.open(data_path).convert("RGB")
        if self.transform:
            data_image = self.transform(data_image)

        label_image = Image.open(label_path).convert("RGB")
        if self.target_transform:
            label_image = self.target_transform(label_image)
        return data_image, label_image

    def __len__(self):
        """
        �������ݼ�����
        :return:
        """
        return len(self.data)


if __name__ == '__main__':
    file_path = './data/ȥ�����ݼ�'  # ���ݼ���Ŀ¼
    data_transform = ToTensor()  # ���ݴ�����
    label_transform = ToTensor()  # ��ǩ������

    nyu2 = NYU2(file_path, transform=data_transform, target_transform=label_transform)
    print(len(nyu2))
    dataloader = DataLoader(nyu2, batch_size=2)
    for X, y in dataloader:
        print(X)
        print(y)
