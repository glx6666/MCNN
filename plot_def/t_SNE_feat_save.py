import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data
from sklearn.manifold import TSNE
import torch.backends.cudnn as cudnn
from ulit.init_seed import *
from ulit.CWRU import *
from ulit.acc import *
import csv
import matplotlib.pyplot as plt
import random
import numpy
from model.MACNN import *

parser = argparse.ArgumentParser(description='PyTorch PN_Data Training')
parser.add_argument('--data', metavar='DIR', default=r'..\dataset\CWRU',help='path to dataset')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default = 64, type=int,metavar='N',help='mini-batch size (default: 128)')
parser.add_argument('-seed', default=123, type=int)
parser.add_argument('-use_model', default='MACNN', help='MACNN')
parser.add_argument('--save_or_not', default=True, type=bool)
parser.add_argument('--result_dir', default=r'..\result', type=str, help='result_root')
parser.add_argument('--save_dir', default=r'..\result', type=str, help='save_root')

def main():
    args = parser.parse_args()
    # 判断是否含有gpu，否则加载到cpu
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("use gpu")
    else:
        device = torch.device('cpu')
        print("use cpu")
    init_seed(args.seed)  # 初始化随机种子参数
    # 构建模型
    if args.use_model == 'MACNN':
        model = MACNN().to(device)
    args.save_dir = os.path.join(args.result_dir, args.use_model)  # 保存路径拼接
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.result_dir = os.path.join(args.save_dir, 'model_50.pth')  # 训练模型参数文件pth，路径拼接
    cwru_data = CWRU(args.data, test_size=0.2)
    _, test_dataset = cwru_data.train_test_split_order()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=True)
    checkpoint = torch.load(args.result_dir)    # 加载训练参数
    model.load_state_dict(checkpoint)  # 模型加载训练参数
    criterion = nn.CrossEntropyLoss().to(device)  # 交叉熵损失函数
    # 测试输出 测试集原始特征分布、全连接层特征分布
    test_feat_save, fc_feat_save, label_save = test(test_loader, model, criterion, device)
    for row in range(0, len(test_feat_save)):
        a1 = test_feat_save[row]
        if row == 0:
            b1 = a1
        else:
            b1 = np.vstack((b1, a1))

    for row in range(0, len(fc_feat_save)):
        a2 = fc_feat_save[row]
        if row == 0:
            b2 = a2
        else:
            b2 = np.vstack((b2, a2))

    for row in range(0, len(label_save)):
        n = label_save[row]
        if row == 0:
            m = n
        else:
            m = np.vstack((m, n))

    test_feat_save_list = b1.T  # 转置
    fc_feat_save_list = b2.T  # 转置
    label_save_list = m.T  # 转置

    if args.save_or_not:
        # 保存特征向量至制定文件夹
        np.savetxt(os.path.join(os.getcwd(), args.save_dir + '\\测试集数据分布.txt'), np.column_stack(test_feat_save_list), delimiter=' ')
        np.savetxt(os.path.join(os.getcwd(), args.save_dir + '\\全连接层特征.txt'), np.column_stack(fc_feat_save_list), delimiter=' ')
        np.savetxt(os.path.join(os.getcwd(), args.save_dir + '\\测试集标签.txt'), np.column_stack(label_save_list), delimiter=' ')

    tsne = TSNE(n_components=2)  # 降维
    labels = np.array(label_save).reshape(1, -1)[0]
    features_tsne_data = tsne.fit_transform(test_feat_save_list.T)
    features_tsne_fc = tsne.fit_transform(fc_feat_save_list.T)
    label_mapping = {0:'Normal', 1:'IR007',  2:'B007',  3:'OR007@6',
                     4:'IR014',  5:'B014',  6:'OR014@6',  7:'IR021',
                     8:'B021',  9:'OR021@6'}
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 绘制二维散点图，根据不同类别用不同颜色标记数据点
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'top': 0.85})
    axes[0].set_title('测试集原始特征分布')
    axes[0].set_xlabel('特征向量X')
    axes[0].set_ylabel('特征向量Y')
    for class_label in np.unique(label_save_list):
        indices = labels == class_label
        chinese_label = label_mapping[class_label]
        axes[0].scatter(features_tsne_data[indices, 0], features_tsne_data[indices, 1],
                    label=f'{chinese_label}')
    axes[1].set_title('全连接层特征分布')
    axes[1].set_xlabel('特征向量X')
    axes[1].set_ylabel('特征向量Y')
    for class_label in np.unique(label_save_list):
        indices = labels == class_label
        chinese_label = label_mapping[class_label]
        axes[1].scatter(features_tsne_fc[indices, 0], features_tsne_fc[indices, 1],
                    label=f'{chinese_label}')
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1), fancybox=True, fontsize='small',
                  ncol=1)
    plt.show()

def test(test_loader, model, criterion, device):
    losses = AverageMeter('Loss', ':.4f')
    raw_feat_list, fc_feat_list, label_list = [], [], []  # 原始数据特征，全连接层特征，标签
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data_raw = data  # 原始信号
            data_raw_feat = data_raw.reshape(data_raw.shape[0], -1)  # 输入电流数据转换维度以便可视化
            input = data.to(device)  # 数据加载到GPU
            train_label = label.to(device) # 标签加载到GPU
            save_feat_raw = model(input)  # 模型输出
            save_feat = save_feat_raw.to("cpu").numpy()  # 特征放到cpu
            label = label.to("cpu").numpy()  # 标签放到cpu
            # 添加训练特征和标签
            raw_feat_list.append(data_raw_feat)
            fc_feat_list.append(save_feat)
            label_list.append(label)
            # 输出损失
            loss = criterion(save_feat_raw, train_label)
            losses.update(loss.item(), train_label.size(0))
        print(f'test_Loss:{losses.avg:.3f}')
        return raw_feat_list, fc_feat_list, label_list

if __name__ == '__main__':
    main()
