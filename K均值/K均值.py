from sklearn.cluster import k_means
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
import pandas as pd

# 导入数据
data = pd.read_csv("three_class_data.csv", header=0)	# 从预先下载的three_class_data.csv读入所有数据
x = data[["x", "y"]]									# 取x、y坐标的2列，也就是输入了一堆二维特征向量。

score = []	# 初始化

for i in range(10):								# 依次计算 2 到 12 类的轮廓系数
    model = k_means(x, n_clusters=i + 2) 		# 算出分2-11类时的K均值。
    score.append(silhouette_score(x, model[1])) # 算出分2-11类时各自的轮廓系数。

plt.subplot(1, 2, 1)				# 画2个图中的画左边的
plt.scatter(data['x'], data['y'])	# 左边画二维特征向量的散点图，也就是把所有二维特征向量作为二维坐标系的点画出来。

plt.subplot(1, 2, 2)				# 画2个图中的画右边的
plt.plot(range(2, 12, 1), score)	# 右边图以有几类（K值）为横坐标，轮廓函数为纵坐标
plt.show()                			# 把画的图显示出来。