# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:51:45 2019

@author: us
"""

'''
k-means是聚类算法，属于非监督学习，
喂给它的数据集是无label的数据，是杂乱无章的，经过聚类后才变得有点顺序，
先无序，后有序
K的含义：K是人工固定好的数字，假设数据集合可以分为K个簇，由于是依靠人工定好，需要一点先验知识

KNN是分类算法，属于监督学习
喂给它的数据集是带label的数据，已经是完全正确的数据
K的含义：来了一个样本x，要给它分类，即求出它的y，
       就从数据集中，在x附近找离它最近的K个数据点，
       这K个数据点，类别c占的个数最多，就把x的label设为c
       
      
'''