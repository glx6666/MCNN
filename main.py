import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import math
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from ulit.init_seed import *
from ulit.acc import *
from ulit.CWRU import *
from model.MACNN import *
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import csv
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

import numpy
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


parser = argparse.ArgumentParser(description='PyTorch PN_Data Training')
parser.add_argument('--data', metavar='DIR', default=r'.\dataset\CWRU_GAF', help='path to dataset')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default = 64, type=int,metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('-gamma', default=0.1, type=float)
parser.add_argument('-stepsize', default=10, type=int)
parser.add_argument('-seed', default=123, type=int)
parser.add_argument('-use_model', default='MACNN', help='MACNN')
# save
parser.add_argument('--save_model', default=True, type=bool)
parser.add_argument('--save_dir', default=r'.\result', type=str, help='save_root')
parser.add_argument('--save_acc_loss_dir', default=r'.\result', type=str)

def main():
    # 判断是否含有gpu，否则加载到cpu
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("use gpu")
    else:
        device = torch.device('cpu')
        print("use cpu")
    args = parser.parse_args()
    init_seed(args.seed)  # 初始化随机种子参数
    criterion = nn.CrossEntropyLoss().to(device)  # 交叉熵损失
    # 构建模型
    if args.use_model == 'MACNN':
        model = MACNN().to(device)
    args.save_dir = os.path.join(args.save_dir, args.use_model)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)  # 如果文件夹不存在，则创建文件夹
    args.save_acc_loss_dir = os.path.join(args.save_dir, 'train_test_result.csv')
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    lr_scheduler = StepLR(optimizer, gamma=args.gamma, step_size=args.stepsize)
    cwru_data = CWRU(args.data, test_size=0.2)
    train_dataset, test_dataset = cwru_data.train_test_split_order()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)
    # 训练、测试模型
    train_acc_list, train_loss_list, test_acc_list, test_loss_list = [], [], [], []
    fault_classes = ['Normal', 'IR007', 'B007', 'OR007@6', 'IR014', 'B014', 'OR014@6', 'IR021', 'B021', 'OR021@6']  # 故障类别
    for epoch in range(args.start_epoch, args.epochs):
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, device)
        test_acc, test_loss, true_labels, pred_labels = test(test_loader, model, criterion, epoch, device)
        train_acc_list.append(round(train_acc, 4))
        train_loss_list.append(round(train_loss, 4))
        test_acc_list.append(round(test_acc, 4))
        test_loss_list.append(round(test_loss, 4))
        # 保存模型
        if args.save_model:
            if epoch == 49:
                model_name = 'model' + '_' + str(epoch + 1) + '.pth'
                torch.save(model.state_dict(), os.path.join(args.save_dir, model_name))
                cm = confusion_matrix(true_labels, pred_labels)
                # 保存混淆矩阵到文件
                np.savetxt(os.path.join(args.save_dir, 'confusion_matrix.txt'), cm, fmt='%d')
                # 输出分类报告
                report = classification_report(true_labels, pred_labels, target_names=fault_classes)
                print(report)
    # 保存准确率-损失
    if args.save_model:
        with open(args.save_acc_loss_dir, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'train_acc', 'train_loss', 'test_acc', 'test_loss'])  # 写入表头
            for epoch in range(len(train_acc_list)):
                writer.writerow([epoch + 1, train_acc_list[epoch], train_loss_list[epoch], test_acc_list[epoch], test_loss_list[epoch]])
    plt.figure(1, figsize=(10, 4))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.subplot(1, 2, 1)
    # 绘制曲线
    plt.plot(train_acc_list, label='训练准确率', linestyle='-', color='blue')
    plt.plot(test_acc_list, label='测试准确率', linestyle='--', color='red')
    # 添加标签和标题
    plt.xlabel('迭代轮数')
    plt.ylabel('准确率')
    plt.title('训练和测试准确率')
    plt.legend(loc='lower right')  # 左下显示
    plt.subplot(1, 2, 2)
    # 绘制曲线
    plt.plot(train_loss_list, label='训练损失', linestyle='-', color='blue')
    plt.plot(test_loss_list, label='测试损失', linestyle='--', color='red')
    # 添加标签和标题
    plt.xlabel('迭代轮数')
    plt.ylabel('损失')
    plt.title('训练和测试损失')
    plt.legend(loc='upper right')  # 右下显示
    fig_path = os.path.join(args.save_dir, 'acc_loss.png')  # 保存图片
    plt.savefig(fig_path)
    plt.show()
    # 可视化混淆矩阵
    plt.figure(2, figsize=(6, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 修改全局配置
    font_options = {
        'family': 'serif',  # 设置字体
        'serif': 'simsun',  # 设置字体
    }
    plt.rc('font', **font_options)
    sns.heatmap(cm, annot=True, fmt='d', cmap='winter', cbar=False, xticklabels=fault_classes, yticklabels=fault_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('混淆矩阵')
    fig_path = os.path.join(args.save_dir, '混淆矩阵.png')
    plt.savefig(fig_path)
    plt.show()


def train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, device):
    losses = AverageMeter('Loss', ':.4f')
    train_acc = AverageMeter('Acc', ':.4f')
    # switch to train mode
    model.train()
    for i, (data, label) in enumerate(train_loader):
        model.zero_grad()
        optimizer.zero_grad()
        input = data.to(device)
        label = label.to(device)
        # compute output and loss
        output = model(input)
        loss = criterion(output, label.long())
        losses.update(loss.item(), label.size(0))
        # Compute accuracy
        _, predicted = torch.max(output, 1)
        accuracy = (predicted == label).sum().item() / label.size(0)
        train_acc.update(accuracy, label.size(0))
        loss.backward()
        optimizer.step()
    lr_scheduler.step()
    print(f'Epoch:[{epoch}] train_Acc:{train_acc.avg:.4f} train_Loss:{losses.avg:.4f}')
    return train_acc.avg,losses.avg


def test(test_loader, model, criterion, epoch, device):
    losses = AverageMeter('Loss', ':.4f')
    test_acc = AverageMeter('Acc', ':.4f')
    # switch to train mode
    model.eval()
    all_preds = []  # 预测标签
    all_labels = []  # 真实标签
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            losses.update(loss.item(), label.size(0))
            # Compute accuracy
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            accuracy = (predicted == label).sum().item() / label.size(0)
            test_acc.update(accuracy, label.size(0))
        print(f'Epoch:[{epoch}] test_Acc:{test_acc.avg:.4f} test_Loss:{losses.avg:.4f}')
    return test_acc.avg, losses.avg, all_labels, all_preds


if __name__ == '__main__':
    main()
