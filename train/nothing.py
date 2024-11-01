from collections import OrderedDict
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# 创建示例 OrderedDict 'features' 和 'labels'
features = OrderedDict([
    ('A', torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])),
    ('B', torch.tensor([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])),
    ('C', torch.tensor([2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0])),
    ('D', torch.tensor([3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0])),
    ('E', torch.tensor([4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0])),
    ('F', torch.tensor([5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0])),
    ('G', torch.tensor([6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0])),
    ('H', torch.tensor([7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0])),
    ('I', torch.tensor([8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0])),
    ('J', torch.tensor([9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0])),
    ('K', torch.tensor([10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0])),
    ('L', torch.tensor([11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0])),
    ('M', torch.tensor([12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0])),
    ('N', torch.tensor([13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14.0])),
    ('O', torch.tensor([14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.0])),
    ('P', torch.tensor([15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16.0])),
    ('Q', torch.tensor([16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 17.0])),
    ('R', torch.tensor([17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7, 17.8, 17.9, 18.0])),
    ('S', torch.tensor([18.1, 18.2, 18.3, 18.4, 18.5, 18.6, 18.7, 18.8, 18.9, 19.0])),
    ('T', torch.tensor([19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 19.7, 19.8, 19.9, 20.0]))
])

labels = OrderedDict([
    ('A', torch.tensor([0])),
    ('B', torch.tensor([1])),
    ('C', torch.tensor([2])),
    ('D', torch.tensor([3])),
    ('E', torch.tensor([4])),
    ('F', torch.tensor([0])),
    ('G', torch.tensor([1])),
    ('H', torch.tensor([2])),
    ('I', torch.tensor([3])),
    ('J', torch.tensor([4])),
    ('K', torch.tensor([0])),
    ('L', torch.tensor([1])),
    ('M', torch.tensor([2])),
    ('N', torch.tensor([3])),
    ('O', torch.tensor([4])),
    ('P', torch.tensor([0])),
    ('Q', torch.tensor([1])),
    ('R', torch.tensor([2])),
    ('S', torch.tensor([3])),
    ('T', torch.tensor([4]))
])

# 提取 features 和 labels 并转换为 NumPy 数组
features_list = [value.numpy() for value in features.values()]
labels_list = [value.item() for value in labels.values()]

X = np.array(features_list)
y = np.array(labels_list)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)

# 定义颜色映射，确保有5种颜色
colors = ['r', 'g', 'b', 'c', 'm']
cmap = ListedColormap(colors)

# 可视化
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=cmap, alpha=0.6)
plt.colorbar(scatter, ticks=range(5), label='Classes')
plt.title('t-SNE Visualization of Features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
