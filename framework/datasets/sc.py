"""
���������ڶ�ȡSpeech Command���ݼ�
    ����: ¬��׿
    ����˼·:
        ��ȷ����Ҫʹ����Щ����: labels, ���Ա�ǩ���б���: SC.encode()
        ����ѡ��ı�ǩ��ȡ��Ӧ�ļ���·��: read_data()
        ����·���õ����ļ��ı�ǩ: get_label()
        ������ݺ�get_label()���getitem()
    δ���:
        ����Ƶ�ļ�·���ж�ȡ�ļ���ȡ��������
"""

import os

from snn_dataset import T_co
from snn_dataset import SNNDataset


class SC(SNNDataset):
    def __init__(self, path, labels, transform=None, target_transform=None):
        """
        ���ݼ���ʼ��
        :param path: ���ݼ���Ŀ¼
        :param labels: ��Ҫ��Speech Command���ݼ��е�����
        :param transform: �������ݴ�����
        :param target_transform: ��ǩ������
        """
        super(SC, self).__init__(path, transform, target_transform)
        self.labels = labels  # ��ǩ��ֵ
        self.labels_dict = self.encode()  # ��ǩ����
        self.data = self.read_data()

    def encode(self):
        """
        ����ǩ���б���
        :return:
        """
        labels_dict = {}
        for i in range(len(self.labels)):
            labels_dict[self.labels[i]] = i
        return labels_dict

    def read_data(self):
        """
        ��ȡȫ������·��
        :return:
        """
        data_paths = []
        for label in self.labels:
            dir_path = self.path + '/' + label  # ��Ӧ��ǩ������
            files = os.listdir(dir_path)
            for file in files:
                data_path = dir_path + '/' + file  # �ļ�·��
                data_paths.append(data_path)
        return data_paths

    def get_label(self, data_path):
        """
        ���ļ�·����ȡ��ǩ�����Ľ��
        :param data_path:  �ļ�·��: SC/label/�����ļ���
        :return:
        """
        label = data_path.split('/')[-2]
        return self.labels_dict[label]

    def __getitem__(self, index) -> T_co:
        return self.data[index], self.get_label(self.data[index])

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    root_path = './data/SC'
    test_labels = ['zero', 'one', 'two', 'three', 'four', 'five']
    sc = SC(root_path, test_labels)
    print(len(sc))
    print(sc[5000])
