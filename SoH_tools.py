from sklearn.preprocessing import MinMaxScaler
from torch.utils.data.dataset import Dataset
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def create_dataset(excel_index):
    # 读取Excel文件，指定没有列标题
    df = pd.read_excel(f'data/test{excel_index}.xlsx', header=None)

    # 初始化一个空列表来存储结果
    # 获取第六列的第一个元素作为输出数据
    output_data = df.iloc[:, 5].values
    results = []
    i = 0
    # 遍历 DataFrame，按第一列的数值进行分组
    for label, group in df.groupby(0):  # 0是第一列的索引位置
        # 将第2到5列的数据转换为一维行向量
        input_data = group.iloc[:, 1:5].values.flatten()
        
        # 将输入数据和输出数据拼接为一个新的行向量
        row_vector = input_data.tolist() + [output_data[i]]
        
        # 将新的行向量添加到结果列表中
        results.append(row_vector)

        i=i+1


    return results


# 最大池化统一数据长度
class SimpleCNN(nn.Module):
    def __init__(self, num_features, output_size):
        super(SimpleCNN, self).__init__()
        # 假设输入数据的形状为 (batch_size, channels, depth, sequence_length)
        # 其中 sequence_length 是变化的
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.pool = nn.AdaptiveAvgPool2d((num_features, output_size))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # 展平特征向量
        x = x.view(x.size(0), -1)
        return x


def unify_dataset(data):
    '''
    读取数据列表 data，每个列表中的一个元素列表为一次放电周期，该元素列表的最后一个元素为输出数据，输入元素为 4 维列向量
    首先读取列表中的每一个元素，将元素列表转化为 array 类型，分离输入和输出数据，并统一输入数据的形式
    其次合并统一格式后的输入数据和输出数据，作为输出数组的第一个元素，其余以此类推
    '''
    # 定义模型参数
    num_features = 4  # 特征数量，与输入数据的channels一致
    output_size = 200  # 输出特征向量的统一长度

    # 创建模型实例
    model = SimpleCNN(num_features, output_size)  # CNN+池化方法

    data_len = len(data)
    for i in range(data_len):
        data[i] = np.array(data[i])
        if i == 0:
            x_set = np.array(data[i][0:-1])
            y_set = np.array(data[i][-1])
            x_set = torch.from_numpy(x_set).float().reshape(1,1,num_features,-1)  # 为了使用CNN的预处理，如果更换其他变换方法可以删除
            x_set = model(x_set)  # 统一长度函数方法，如果设置其他函数在此修改
            y_set = torch.from_numpy(y_set).float().reshape(1,-1)
            data_set = torch.hstack((x_set, y_set))
            results = data_set
        else:
            x_set = np.array(data[i][0:-1])
            y_set = np.array(data[i][-1])
            x_set = torch.from_numpy(x_set).float().reshape(1,1,num_features,-1)  # 为了使用CNN的预处理，如果更换其他变换方法可以删除
            x_set = model(x_set)  # 统一长度函数方法，如果设置其他函数在此修改
            y_set = torch.from_numpy(y_set).float().reshape(1,-1)
            data_set = torch.hstack((x_set, y_set))
            results = torch.vstack((results, data_set))
        
        return results


def train_scale(train):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(train)
    train_scaled = scaler.fit_transform(train)
    return scaler, train_scaled


def test_scale(test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(test)
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.fit_transform(test)
    return scaler, test_scaled


def invert_scale(scaler: object, X: object, value: object) -> object:
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array) # 去标准化
    inverted = torch.Tensor(inverted)
    return inverted[0, -1]


class DataPrepare(Dataset):

    def __init__(self, train):
        self.len = train.shape[0]
        x_set = train[:, 0:-1]
        x_set = x_set.reshape(x_set.shape[0], 1, 5)
        # 数据类型转为 torch 变量
        self.x_data = torch.from_numpy(x_set)
        self.y_data = torch.from_numpy(train[:, [-1]])


    def __getitem__(self, index):
        # 返回 img, label
        return self.x_data[index], self.y_data[index]
    

    def __len__(self):
        return self.len