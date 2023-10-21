"""
本块代码负责设计数据集的基类
"""
from typing import TypeVar
from torch.utils.data import Dataset

T_co = TypeVar('T_co', covariant=True)


class SNNDataset(Dataset):
    """
    数据集基类,为了适应众多数据集,我们设定数据集中应至少包含以下内容:
        1.数据在哪里(文件夹, 路径文档, .h5文件) => path
        2.怎么读取数据(如何从特定的格式、特定的文件组织中获取数据)和获取对应数据的标签(包括标签编码等) => read_data, get_label
        3.怎么处理数据(例如将图片数据转化为tensor, 涉及到样本数据和样本标签两方面的数据处理) => transform, target_transform
        4.怎么使用数据 => get_item， len(继承torch.utils.data.Dataset需要实现的方法, 用于后续装载仅DataLoader)
    """
    def __init__(self, path, transform=None, target_transform=None):
        """
        :param path: 数据集路径
        :param transform: 样本数据处理函数
        :param target_transform: 标签数据处理函数
        """
        self.path = path
        self.transform = transform
        self.target_transform = target_transform

    def read_data(self):
        raise NotImplementedError

    def get_label(self):
        raise NotImplementedError

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
