from sklearn.preprocessing import MinMaxScaler
from torch.utils.data.dataset import Dataset
import torch
import pandas as pd
import numpy as np


def create_dataset(excel_index):
    df = pd.read_excel('data\data1.xlsx', header=None)

    # 初始化一个空列表来存储结果
    # 获取第 6 列输出数据
    output_data = df.iloc[:, 5].values
    dataset = []
    i = 0
    # 遍历 DataFrame，按第一列的数值进行分组
    for label, group in df.groupby(0):  # 0是第一列的索引位置
        # 将第 2 到 5 列的数据转换为一维行向量
        input_data = group.iloc[:, 1:5].values.flatten()
        
        # 将输入数据和输出数据拼接为一个新的行向量
        row_vector = input_data.tolist() + [output_data[i]]
        
        # 将新的行向量添加到结果列表中
        dataset.append(row_vector)

        i=i+1
    
    return dataset


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