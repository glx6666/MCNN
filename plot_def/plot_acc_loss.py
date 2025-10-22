import matplotlib.pyplot as plt
import pandas as pd
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
acc_or_loss = 'acc'  # 选择测试准确率还是损失，将3个方法的曲线进行对比，出在一张图
save_png_root = '../result'  # 图片保存路径
save_root_MsCNN = os.path.abspath(os.path.join(script_dir, '..', 'result', 'ResNet', 'train_test_result.csv'))
# 读取 CSV 文件
df_MCNN = pd.read_csv(save_root_MsCNN)  # 读取MsCNN CSV 文件路径
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
if acc_or_loss == 'acc':
    MACNN_test_accuracy = df_MCNN.iloc[:, 3]
    plt.figure(figsize=(5, 4))
    # 绘制曲线
    plt.plot(MACNN_test_accuracy, label='ResNet', linestyle='-', color='r', marker='.')
    # 添加标签和标题
    plt.xlabel('迭代轮数')
    plt.ylabel('准确率')
    plt.title('准确率曲线')
    # 显示图例（右上角）lower upper
    plt.legend(loc='lower right')
    # 显示图形
    fig_path = os.path.join(save_png_root, 'acc.png')
    plt.savefig(fig_path)
    plt.show()
else:
    MACNN_test_accuracy = df_MCNN.iloc[:, 4]
    plt.figure(figsize=(5, 4))
    # 绘制曲线
    plt.plot(MACNN_test_accuracy, label='ResNet', linestyle='-', color='r', marker='.')

    # 添加标签和标题
    plt.xlabel('迭代轮数')
    plt.ylabel('损失')
    plt.title('损失曲线')
    # 显示图例（右上角）lower upper
    plt.legend(loc='upper right')
    # 显示图形
    fig_path = os.path.join(save_png_root, 'loss.png')
    plt.savefig(fig_path)
    plt.show()
