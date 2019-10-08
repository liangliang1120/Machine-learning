import scipy
import matplotlib
import numpy as np
import sklearn
from scipy import sparse

Z = np.random.random((5,10))
print(Z)

Z[Z<0.7] = 0
print(Z)
Z_csr = sparse.csr_matrix(Z)  #压缩矩阵 坐标
print(Z_csr)
print(Z_csr.toarray())

from sklearn.datasets import load_iris
iris = load_iris()
iris.keys()
# ['data', 'target', 'target_names', 'DESCR', 'feature_names']
# data是训练数据，是一个矩阵，有两个维度，一个维度表示样本数量，一个是特征数量
# target标注
n_samples, n_features = iris.data.shape
# (150, 4) 150个样本，4个特征
iris.target.shape
# (150,)
iris.feature_names
# ['sepal length (cm)',
#  'sepal width (cm)',
#  'petal length (cm)',
#  'petal width (cm)']
iris.target_names
#  array(['setosa', 'versicolor', 'virginica'], dtype='<U10') 哪种花
import matplotlib.pyplot as plt
x_index = 0
y_index = 1
plt.scatter(iris.data[:,x_index], iris.data[:,y_index],c=iris.target)
plt.xlabel = (iris.feature_names[x_index])
plt.ylabel = (iris.feature_names[y_index])