import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(20230226)  # 固定随机种子
torch.backends.cudnn.deterministic = True  # 固定GPU运算方式
amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
def getSequenceData(direction: str):
    # 从目标路径加载数据
    data, label = [], []
    max_length = 0
    min_length = 8000

    with open(direction) as f:  # 读取文件
        for each in f:  # 循环一：文件中每行内容
            each = each.strip()  # 去除字符串首尾的空格
            each = each.upper()  # 将小写转为大写
            if each[0] == '>':
                label.append(np.array(list(each[1:]), dtype=int))  # Converting string labels to numeric vectors
            else:
                max_length = max(max_length, len(each.split('\n')[0]))  # 序列最大长度
                min_length = min(min_length, len(each.split('\n')[0]))  # 序列最小长度
                data.append(each)
    return np.array(data), np.array(label), max_length, min_length
def PadEncode(data, label, max_len: int = 50):
    # 序列编码
    data_e, label_e, seq_length, temp = [], [], [], []
    sign, b = 0, 0
    for i in range(len(data)):
        length = len(data[i])
        if len(data[i]) > max_len:  # 剔除序列长度大于50的序列
            continue
        element, st = [], data[i].strip()
        for j in st:
            if j not in amino_acids:  # 剔除包含非天然氨基酸的序列
                sign = 1
                break
            index = amino_acids.index(j)  # 获取字母索引
            element.append(index)  # 将字母替换为数字
            sign = 0

        if length <= max_len and sign == 0:  # 序列长度复合要求且只包含天然氨基酸的序列

            temp.append(element)
            seq_length.append(len(temp[b]))  # 保存序列有效长度
            b += 1
            element += [0] * (max_len - length)  # 用0补齐序列长度
            data_e.append(element)
            label_e.append(label[i])

    return torch.LongTensor(np.array(data_e)), torch.LongTensor(np.array(label_e)), torch.LongTensor(
        np.array(seq_length))

def data_load(train_direction=None, test_direction=None, batch=None, subtest=True, CV=False,threshold_percentile=None):
    # 从目标路径加载数据
    dataset_train, dataset_test = [], []
    dataset_subtest = None
    weight = None
    # 加载数据
    train_seq_data, train_seq_label, max_len_train, min_len_train = getSequenceData(train_direction)
    test_seq_data, test_seq_label, max_len_test, min_len_test = getSequenceData(test_direction)
    print(f"max_length_train:{max_len_train}")
    print(f"min_length_train:{min_len_train}")
    print(f"max_length_test:{max_len_test}")
    print(f"min_length_test:{min_len_test}")

    x_train, y_train, train_length = PadEncode(train_seq_data, train_seq_label, max_len_train)
    x_test, y_test, test_length = PadEncode(test_seq_data, test_seq_label, max_len_test)
    # 计算类别权重
    if CV is False:  # 不进行五折交叉验证
        # Create datasets
        train_data = TensorDataset(x_train, train_length, y_train)
        test_data = TensorDataset(x_test, test_length, y_test)
        dataset_train.append(DataLoader(train_data, batch_size=batch, shuffle=True))
        dataset_test.append(DataLoader(test_data, batch_size=batch, shuffle=True))

        # 构造测试子集
        if subtest:
            dataset_subtest = []
            for i in range(5):  # 从测试集中随机抽取80%作为子集，重复5次得到5个子集
                sub_size = int(0.8 * len(test_data))
                _ = len(test_data) - sub_size
                subtest, _ = torch.utils.data.random_split(test_data, [sub_size, _])
                sub_test = DataLoader(subtest, batch_size=batch, shuffle=True)
                dataset_subtest.append(sub_test)
    else:
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        # 构造五折交叉验证的训练集和测试集
        for split_index, (train_index, test_index) in enumerate(cv.split(x_train)):
            sequence_train, label_train, length_train = x_train[train_index], y_train[train_index], \
                                                        train_length[train_index]
            sequence_test, label_test, length_test = x_train[test_index], y_train[test_index], train_length[
                test_index]
            train_data = TensorDataset(sequence_train, length_train, label_train)
            test_data = TensorDataset(sequence_test, length_test, label_test)
            dataset_train.append(DataLoader(train_data, batch_size=batch, shuffle=True))
            dataset_test.append(DataLoader(test_data, batch_size=batch, shuffle=True))
    return dataset_train, dataset_test, dataset_subtest, weight

