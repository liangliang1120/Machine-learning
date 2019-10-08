import numpy as np
from numpy import array,shape,arange
import matplotlib.pyplot as plt

# plt.switch_backend('agg')
# %matplotlib inline

import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
# LassoCV 交叉验证
from sklearn.linear_model import RidgeCV, Ridge
from sklearn import datasets
from sklearn.model_selection import train_test_split
# train_test_split=把数据分解成训练集和测试集，为了进行交叉验证，选择超参数和正则化的变量，评估结构风险
# 我们要选择test_size，控制测试集和训练集的比例，我们要选择random seed，来控制 随机种子在其他情况，让random seed取不同的数
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets,linear_model

diabetes = datasets.load_diabetes()
# diabetes 数据，从sklearn学出来的
X = diabetes.data  # 观测变量
y = diabetes.target  # 预测值

SEED = 30
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state=SEED)
diabetes= datasets.load_diabetes()
indices = (0, 1)

X_train = diabetes.data[:-20, indices]
X_test = diabetes.data[-20:, indices]
y_train = diabetes.target[:-20]
y_test = diabetes.target[-20:]

ols = linear_model.LinearRegression()
ols.fit(X_train, y_train)

def plot_figs(fig_num, elev, azim, X_train, clf):
    fig = plt.figure(fig_num, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, elev=elev, azim=azim)

    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='k', marker='+')
    fake_data = np.zeros((4, X_train.shape[1]))
    fake_data[:, :2] = np.array([[-.1, -.1, .15, .15], [-.1, .15, -.1, .15]]).T

    ax.plot_surface(np.array([[-.1, -.1], [.15, .15]]),
                    np.array([[-.1, .15], [-.1, .15]]),
                    clf.predict(fake_data).reshape((2, 2)), alpha=.5)
    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_zlabel('Y')
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

ols = linear_model.LinearRegression()
ols.fit(X_train, y_train)

elev = 43.5
azim = -110
plot_figs(1, elev, azim, X_train, ols)

elev = -.5
azim = 0
plot_figs(2, elev, azim, X_train, ols)

elev = -.5
azim = 90
plot_figs(3, elev, azim, X_train, ols)
plt.show()

# print(diabetes)

