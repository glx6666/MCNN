import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


script_dir = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.abspath(os.path.join(script_dir, '..', 'result', 'MACNN'))
feature_select = '全连接层特征.txt'#全连接层特征.txt,测试集数据分布.txt
test_feat_path = os.path.abspath(os.path.join(script_dir, save_path, feature_select))  # 测试集数据分布,全连接层特征
label_path = os.path.abspath(os.path.join(script_dir, save_path, '测试集标签.txt'))  # 标签路径拼接
# 读取特征向量
test_feat = np.loadtxt(test_feat_path, delimiter=' ')
labels = np.loadtxt(label_path, delimiter=' ')

# 标签映射表
label_mapping = {0: 'Normal', 1: 'IR007', 2: 'B007', 3: 'OR007@6',
                 4: 'IR014', 5: 'B014', 6: 'OR014@6', 7: 'IR021',
                 8: 'B021', 9: 'OR021@6'}
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# T-SNE降维并保存降维的特征，因为T-SNE每次运行后降维特征不一致，第一次运行需要保存降维特征，后面就只用读取保存的降维特征进行可视化
tsne_test_feat = TSNE(n_components=2).fit_transform(test_feat)
labels = np.array(labels).reshape(1, -1)[0]
if feature_select == '全连接层特征.txt':
    np.savetxt(os.path.join(os.getcwd(), save_path + '\\全连接层特征T_SNE降维.txt'), np.column_stack(tsne_test_feat),delimiter=' ')
else:
    np.savetxt(os.path.join(os.getcwd(), save_path + '\\测试集特征T_SNE降维.txt'), np.column_stack(tsne_test_feat),delimiter=' ')

plt.figure(figsize=(5, 4))
unique_labels = np.unique(labels)
for class_label in unique_labels:
    indices = labels == class_label
    plt.scatter(tsne_test_feat[indices, 0], tsne_test_feat[indices, 1], label=f'Class {class_label}')
legend_labels = [label_mapping[label] for label in unique_labels]
plt.legend(legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize='small', fancybox=True, ncol=len(unique_labels)/2)
# Save the figure
plt.xlabel('特征向量X')
plt.ylabel('特征向量Y')
if feature_select == '全连接层特征.txt':
    fig_path = os.path.join(save_path, '全连接层特征分布图.png')
    plt.savefig(fig_path)
else:
    fig_path = os.path.join(save_path, '测试集特征分布图.png')
    plt.savefig(fig_path)
plt.show()
