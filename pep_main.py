
import argparse
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
    parse.add_argument('--loss_weights', nargs='+', type=float, default=[0.99,0.01],
                       help='Weights for each loss function. Must match the number of criterion. '
                            'e.g., --loss_weights 0.7 0.3 0.3 0.2')
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
    parse.add_argument('-threshold_percentile', type=int, default=40)
    config = parse.parse_args()  # 解析所添加的参数
    return config

if __name__ == '__main__':
    args = get_config()                  #从配置文件中获取参数
    from train_test import TrainAndTest
    TrainAndTest(args)


