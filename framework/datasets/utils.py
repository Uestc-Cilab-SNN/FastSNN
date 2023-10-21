"""
本块代码负责设计数据集部分的工具函数:
    读取数据集(启发式的, 用于告诉用户如何读取特定格式的数据集, 大概率需要自定义): .h5, dir, txt, gz
"""
import h5py
import os


def read_h5(path, data_key=None, label_key=None):
    """
    如果数据集是按照.h5(HDF5)的格式存储的:
        使用h5py库来打开.h5文件:  f = h5py.File(path, 'r')
        根据key(多级目录结构)来获取对应部分的数据: data = f[key_1][key_2][:] => numpy.ndarray
    部分数据集是直接将数据和标签显式地存储在.h5中的, 本方法便是针对这类型数据集(作为示例)
    然而各个数据集的标准不同, 还有很多数据集可能key同时存在.h5中, 不能显式地使用key来读取数据(n-tidigits)
    或者是可能只有数据在.h5中, 而标签使用其它的方式存储, 需要后续针对性地自定义
    :param path: .h5文件路径
    :param data_key: list 数据索引(根据具体的目录结构来分析)
    :param label_key: list 标签索引(根据具体的目录结构来分析)
    :return:
    """
    f = h5py.File(path, 'r')  # 读取文件
    data = f
    label = f
    if data_key:  # 如果是知道数据索引的
        for key in data_key:
            data = data[key]
        data = data[:]

    if label_key:  # 如果是知道标签索引的
        for key in label_key:
            label = label[key]
        label = label[:]
    return data, label


def read_dir(path):
    """
    如果数据集是存放在某个文件夹下的:
        我们人为规定数据存放在path/data/ 文件夹下
        然而实际上数据和标签存放的路径和方法不尽相同, 请自觉酌情更改
    该方法需要做的是获取数据的地址(本质上与路径文档相同)并返回(而不是直接保存数据再返回)
    本部分代码仅包含对于样本数据的读取,不包含对于标签的读取(标签的保存和获取方法多种多样, 难以涵盖全部)
    :param path: 根文件夹路径
    :return:
    """
    data_path = path + '/data'  # 数据文件夹路径
    data = os.listdir(data_path)  # 获取文件夹下所有的文件名
    for i in range(len(data)):
        data[i] = data_path + '/' + data[i]  # 构造数据路径
    return data


def read_txt(path):
    """
    如果数据集的数据地址存放在某个文本文档中, 那么该文本文档应符合如下要求:
        简单明了的编码方式: gbk, utf-8等
        文档中每一行(每一个路径)表示一个样本数据, 每一行路径字符串后应接换行符\n, 并在读取时将其消除
    最终返回文本文档中存储的数据路径(不含标签)
    :param path:
    :return:
    """
    data_paths = []
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readline()
        while data:
            data_paths.append(data.replace('\n', ''))
            data = f.readline()
    return data_paths

