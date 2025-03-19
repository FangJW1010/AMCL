from models.TextCNN import TextCNN
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import csv
import torch
import sys
import os
import pandas as pd
import numpy as np
import estimate
import argparse
torch.manual_seed(20230226)  # 固定随机种子
torch.backends.cudnn.deterministic = True  # 固定GPU运算方式
amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
import logging
import torch
import time
def create_logger(log_name='test.log', mode='a'):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # 创建控制台handler并设置级别
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # 创建文件handler并设置级别
    file_handler = logging.FileHandler(log_name, mode=mode)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
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

sys.path.append('C:\\Users\\fly\\Desktop\\')

torch.manual_seed(20230226)  # 设置CPU的随机种子
torch.cuda.manual_seed(20230226)  # 设置GPU的随机种子
torch.backends.cudnn.deterministic = True  # 固定GPU运算方式
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备
# 数据集中治疗肽的种类
RMs = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP',
       'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP', 'AVP', 'BBP',
       'BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP']

# 保存运行结果的csv中的标题
title1 = ['Model', "Loss", 'Aiming', 'Coverage', 'Accuracy',
          'Absolute_True', 'Absolute_False', 'RunTime']

def spent_time(start, end):
    # 计算代码的运行时间
    epoch_time = end - start
    minute = int(epoch_time / 60)  # 分钟
    secs = int(epoch_time - minute * 60)  # 秒
    return minute, secs


def save_results(model_name, loss_name, start, end, test_score, class_scores, file_path):
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    content = [[model_name, loss_name,
                '%.3f' % test_score[0],
                '%.3f' % test_score[1],
                '%.3f' % test_score[2],
                '%.3f' % test_score[3],
                '%.3f' % test_score[4],
                '%.3f' % (end - start),
                now]]

    class_title = ['Class', 'TP', 'FP', 'TN', 'FN']
    class_results = []

    for i, rm in enumerate(RMs):
        class_results.append([rm, class_scores[0][i], class_scores[1][i], class_scores[2][i], class_scores[3][i]])

    if os.path.exists(file_path):
        try:
            data = pd.read_csv(file_path, encoding='gbk')
            with open(file_path, 'a+', newline='') as t:
                writer = csv.writer(t)
                writer.writerows(content)
                writer.writerow([])
                writer.writerow(['Model'] + class_title)
                writer.writerows(class_results)

        except Exception as e:
            print(f"Error reading existing CSV file: {e}. Creating new file.")
            with open(file_path, 'w', newline='') as t:
                writer = csv.writer(t)
                writer.writerow(
                    ['Model', 'Loss', 'Aiming', 'Coverage', 'Accuracy', 'Absolute_True', 'Absolute_False', 'Runtime',
                     'Timestamp'])
                writer.writerows(content)
                writer.writerow([])  # Add an empty row for separation
                writer.writerow(['Model'] + class_title)  # Use consistent header
                writer.writerows(class_results)
    else:
        with open(file_path, 'w', newline='') as t:
            writer = csv.writer(t)
            writer.writerow(
                ['Model', 'Loss', 'Aiming', 'Coverage', 'Accuracy', 'Absolute_True', 'Absolute_False', 'Runtime',
                 'Timestamp'])
            writer.writerows(content)
            writer.writerow([])
            writer.writerow(['Model'] + class_title)
            writer.writerows(class_results)

Train_path = "dataset/augmented_train.txt"
class_freq = [0] * 21  # Initialize counts for each peptide
train_num=0
with open(Train_path, 'r') as file:
    for line in file:
        if line.startswith('>'):  # Assume each line with data starts with '>'
            train_num+=1
            line = line.strip()[1:]  # Remove the '>' and any whitespace
            for i, char in enumerate(line):
                if char == '1':
                    class_freq[i] += 1

def gain_class_freq_and_train_num():
    return class_freq,train_num

def custom_sigmoid_multi_threshold(x, thresholds, k=10):    #k=1
    """支持多阈值的sigmoid函数"""
    thresholds = thresholds.view(1, -1)
    return 1 / (1 + torch.exp(-k * (x - thresholds)))

def predict_with_thresholds(model, data, thresholds, device="cuda", return_features=False):
    model.to(device)
    model.eval()
    all_predictions = []
    all_labels = []
    all_features = []

    thresholds = torch.tensor(thresholds).to(device)

    with torch.no_grad():
        for batch in data:
            x, l, y = batch
            x = x.to(device)

            if return_features:
                scores, features = model(x, return_features=True)
                all_features.extend(features.cpu().numpy())
            else:
                scores = model(x, return_features=False)

            # 使用sigmoid函数
            sigmoid_scores = custom_sigmoid_multi_threshold(scores, thresholds)
            predictions = torch.zeros_like(scores)
            for i in range(scores.shape[1]):
                # 使用sigmoid后的分数与阈值比较
                predictions[:, i] = (sigmoid_scores[:, i] > thresholds[i]).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels)

def evaluate_with_thresholds(predictions, labels):
    """评估二值化后的预测结果"""
    absolute_true = 0
    total_samples = len(predictions)

    # 计算完全匹配的样本数
    for i in range(total_samples):
        if np.array_equal(predictions[i], labels[i]):
            absolute_true += 1
    absolute_true_rate = absolute_true / total_samples
    other_metrics = list(estimate.evaluate(predictions, labels, threshold=None))
    metrics = (
        other_metrics[0],  # aiming
        other_metrics[1],  # coverage
        other_metrics[2],  # accuracy
        absolute_true_rate,  #absolute_true
        other_metrics[3]  # absolute_false
    )
    return metrics
def optimize_thresholds_with_freq(model, val_data, class_freq, train_num, device="cuda", init_thresholds=None):
    """考虑类别频率的阈值优化"""
    init_thresholds = [0] * 21
    predictions = []
    true_labels = []

    model.eval()
    with torch.no_grad():
        for batch in val_data:
            x, l, y = batch
            x = x.to(device)
            score = model(x, return_features=False)
            predictions.append(score.cpu())
            true_labels.append(y.cpu())

    predictions = torch.cat(predictions, dim=0).numpy()
    true_labels = torch.cat(true_labels, dim=0).numpy()
    class_ratios = [freq / train_num for freq in class_freq]
    optimal_thresholds = []

    # 对每个类别优化阈值
    for i in range(21):
        ratio = class_ratios[i]
        class_preds = predictions[:, i]
        class_labels = true_labels[:, i]
        # 设置搜索范围
        if ratio < 0.005:
            threshold_range = np.arange(0.1, 0.3, 0.01)
        elif ratio < 0.02:
            threshold_range = np.arange(0.3, 0.5, 0.01)
        elif ratio < 0.05:
            threshold_range = np.arange(0.4, 0.6, 0.01)
        elif ratio < 0.08:
            threshold_range = np.arange(0.45, 0.65, 0.01)
        elif ratio < 0.15:
            threshold_range = np.arange(0.5, 0.7, 0.01)
        else:
            threshold_range = np.arange(0.55, 0.8, 0.01)

        best_threshold = init_thresholds[i]
        best_score = float('-inf')
        # 尝试不同阈值
        for threshold in threshold_range:
            pred_labels = (class_preds > threshold).astype(float)
            current_scores = estimate.evaluate(
                pred_labels.reshape(1, -1),
                class_labels.reshape(1, -1),
                threshold=None
            )
            # 根据类别频率调整评分权重
            if ratio < 0.05:
                combined_score = (
                        current_scores[0] * 0.3 +
                        current_scores[1] * 0.1 +
                        current_scores[2] * 0.6
                )
            else:
                combined_score = (
                        current_scores[0] * 0.4 +
                        current_scores[1] *0.3 +
                        current_scores[2] * 0.3
                )

            if combined_score > best_score:
                best_score = combined_score
                best_threshold = threshold

        optimal_thresholds.append(best_threshold)
        print(f"Class {i}: ratio = {ratio:.4f}, threshold = {best_threshold:.3f}")

    return optimal_thresholds
def get_per_class_metrics(predictions, true_labels):

    num_classes = predictions.shape[1]
    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []

    for i in range(num_classes):
        TP = ((predictions[:, i] == 1) & (true_labels[:, i] == 1)).sum().item()
        FP = ((predictions[:, i] == 1) & (true_labels[:, i] == 0)).sum().item()
        TN = ((predictions[:, i] == 0) & (true_labels[:, i] == 0)).sum().item()
        FN = ((predictions[:, i] == 0) & (true_labels[:, i] == 1)).sum().item()

        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)

    return [TP_list, FP_list, TN_list, FN_list]

def get_config():
    # 参数管理
    parse = argparse.ArgumentParser(description='peptide default main')#创建了一个 ArgumentParser 对象 parse，作为命令行解析器，并设置了该解析器的描述。
    parse.add_argument('-task', type=str, default='model_select')#添加一个名为 -task 的参数，并设置其默认值为 'model_select'。
    # 模型训练参数
    parse.add_argument('-model', type=str, default='TextCNN',
                       help='The name of model')
    parse.add_argument('--criterion', nargs='+', type=str, default=['FDL','DBL'],
                       help='{CL: Combo loss, FDL: Focal dice loss, DL: Dice loss, CE: Cross entropy loss, '
                            'ASL: Asymmetric loss, FL: Focal loss} ')
    parse.add_argument('-subtest', type=bool, default=False)
    parse.add_argument('-vocab_size', type=int, default=21,
                       help='The size of the vocabulary')
    parse.add_argument('-output_size', type=int, default=21,
                       help='Number of peptide functions') #多肽数量
    parse.add_argument('-batch_size', type=int, default=64*4,
                       help='Batch size')
    parse.add_argument('-epochs', type=int, default=200)
    parse.add_argument('-learning_rate', type=float, default=0.0007)
    # 深度模型参数
    parse.add_argument('-embedding_size', type=int, default=64*2,
                       help='Dimension of the embedding')
    parse.add_argument('-dropout', type=float, default=0.6)
    parse.add_argument('-filter_num', type=int, default=64*2,
                       help='Number of the filter') #滤波器数量
    parse.add_argument('-filter_size', type=list, default=[3,4,5],
                       help='Size of the filter')
    # 文件路径
    parse.add_argument('-model_path', type=str, default='saved_models/model_select+TextCNN0.pth',
                       help='Path of the training data')
    parse.add_argument('-train_direction', type=str, default='dataset/augmented_train.txt',
                       help='Path of the training data')
    parse.add_argument('-test_direction', type=str, default='dataset/test.txt',
                       help='Path of the test data')
    config = parse.parse_args()  # 解析所添加的参数
    return config

if __name__ == '__main__':
    args = get_config()   #从配置文件中获取参数
    def TrainAndTest(args):
        logger = create_logger()
        logger.info(f"This task is {args.task}")
        models_file = f'result/{args.task}_models.txt'
        Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        parse_file = f"result/{args.task}_pares.txt"
        with open(parse_file, 'a') as file1:
            file1.write(Time + '\n')
            print(args, file=file1)
            file1.write('\n')

        file_path = f'result/model_select.csv'
        logger.info("Data is loading ......（￣︶￣）↗　")

        train_datasets, test_datasets, subtests, weight = data_load(
            batch=args.batch_size,
            train_direction=args.train_direction,
            test_direction=args.test_direction,
            subtest=args.subtest,
            CV=False
        )
        logger.info("Data is loaded!ヾ(≧▽≦*)o")
        class_freq, train_num = gain_class_freq_and_train_num()
        start_time = time.time()
        for i in range(len(train_datasets)):
            train_dataset = train_datasets[i]
            test_dataset = test_datasets[i]
            train_start = time.time()

            # 初始化模型
            if args.model == 'TextCNN':
                model = TextCNN(args.vocab_size, args.embedding_size, args.filter_num,
                                args.filter_size, args.output_size, args.dropout)
            else:
                raise Exception('Unexpected model {}'.format(args.model))

            model_name = model.__class__.__name__
            title_task = f"{args.task}+{model_name}"

            # 记录模型结构
            model_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open(models_file, 'a') as file2:
                file2.write(model_time + '\n')
                print(model, file=file2)
                file2.write('\n')
            try:
                model.load_state_dict(torch.load(args.model_path))
                model = model.to(DEVICE)
                loss_name = "FocalDiceLoss+ResampleLoss"
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
            # 优化阈值
            logger.info("Optimizing thresholds...")
            optimal_thresholds = optimize_thresholds_with_freq(
                model,
                train_dataset,
                class_freq,
                train_num,
                device=DEVICE
            )
            # 使用优化后的阈值进行预测和评估
            model_predictions, true_labels = predict_with_thresholds(
                model,
                test_dataset,
                thresholds=optimal_thresholds,
                device=DEVICE
            )
            # 评估结果
            test_score = evaluate_with_thresholds(model_predictions, true_labels)

            # 获取每个类的TP、FP、TN、FN
            class_scores = get_per_class_metrics(model_predictions, true_labels)

            # 保存结果
            test_end = time.time()
            save_results(title_task, loss_name, train_start, test_end, test_score, class_scores, file_path)
            # 打印评估结果
            run_time = time.time()
            m, s = spent_time(start_time, run_time)
            logger.info(f"{args.task}, {model_name}'s runtime:{m}m{s}s")
            logger.info("Test Results:")
            logger.info(f'Aiming: {test_score[0]:.3f}')
            logger.info(f'Coverage: {test_score[1]:.3f}')
            logger.info(f'Accuracy: {test_score[2]:.3f}')
            logger.info(f'Absolute True: {test_score[3]:.3f}')
            logger.info(f'Absolute False: {test_score[4]:.3f}\n')
    TrainAndTest(args)


