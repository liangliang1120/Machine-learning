import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn import preprocessing

import numpy as np
x=np.arange(10).reshape(5,2)

rfm_data = pd.read_csv('RFM.csv')
rfm_init = rfm_data[['userid','Recency_y','Frequency_y','Monetary_y']]
# 采用Z-Score规范化
ss = preprocessing.StandardScaler()
rfm_np = rfm_init[['Recency_y','Frequency_y','Monetary_y']].values
train_ss_x = ss.fit_transform(rfm_np)

cluster = KMeans(n_clusters=8, max_iter=500)
a = cluster.fit(train_ss_x)
b = cluster.cluster_centers_
c = cluster.labels_
rfm_data['kmeans_kind'] = c

#============画图==========================================================================
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

r = rfm_init['Recency_y'].values
f = rfm_init['Frequency_y'].values
m = rfm_init['Monetary_y'].values
def plot_figs(fig_num, elev, azim, X_train, y_train, z_train):
    fig = plt.figure(fig_num, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, elev=elev, azim=azim)

    ax.scatter(X_train, y_train, z_train, c='k', marker='+')

    ax.plot_surface(np.array([[0, 100], [0, 100]]),
                    np.array([[0, 60], [0, 60]]),
                    np.array([[0, 240000], [0, 240000]]), alpha=.5)
    ax.set_xlabel('r-Label')
    ax.set_ylabel('f-Label')
    ax.set_zlabel('m-Label')

    '''
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    '''
elev = 43.5
azim = -110
plot_figs(1, elev, azim, r, f, m)

r_v = rfm_data['R'].values
f_v = rfm_data['F'].values
m_v = rfm_data['M'].values
def plot_figs(fig_num, elev, azim, X_train, y_train, z_train):
    fig = plt.figure(fig_num, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, elev=elev, azim=azim)

    ax.scatter(X_train, y_train, z_train, c='k', marker='+')

    ax.plot_surface(np.array([[0, 0], [4, 4]]),
                    np.array([[0, 0], [4, 4]]),
                    np.array([[2, 2], [2, 2]]), alpha=.5)
    ax.set_xlabel('r-score')
    ax.set_ylabel('f-score')
    ax.set_zlabel('m-score')

    '''
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    '''
elev = 43.5
azim = -110
plot_figs(1, elev, azim, r_v, f_v, m_v)



