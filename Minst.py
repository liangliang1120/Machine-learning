# -*- coding: utf-8 -*-
# 使用LR进行MNIST手写数字分类
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 加载数据
digits = load_digits()
data = digits.data
# 数据探索
print(data.shape)
# 查看第一幅图像
print(digits.images[0])
# 第一幅图像代表的数字含义
print(digits.target[0])
# 将第一幅图像显示出来
plt.gray()
plt.imshow(digits.images[0])
plt.show()

# 分割数据，将25%的数据作为测试集，其余作为训练集
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)

# 采用Z-Score规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

# 创建LR分类器
lr = LogisticRegression()
lr.fit(train_ss_x, train_y)
predict_y=lr.predict(test_ss_x)
print('LR准确率: %0.4lf' % accuracy_score(predict_y, test_y))

'''
逻辑回归由于存在易于实现、解释性好以及容易扩展等优点，被广泛应用于点击率预估（CTR）、计算广告（CA）以及推荐系统（RS）等任务中。逻辑回归虽然名字叫做回归，但实际上却是一种分类学习方法。 
预测函数

对于二分类问题，y∈{0,1}y∈{0,1}，1表示正例，0表示负例。逻辑回归是在线性函数θTx输出预测实际值的基础上，寻找一个假设函数函数hθ(x)=g(θTx)，将实际值映射到到0，1之间，如果hθ(x)>=0.5hθ(x)>=0.5，则预测y=1y=1，及yy属于正例；如果hθ(x)<0.5hθ(x)<0.5，则预测y=0y=0，即yy属于负例。

逻辑回归中选择对数几率函数（logistic function）作为激活函数，对数几率函数是Sigmoid函数（形状为S的函数）的重要代表

逻辑回归是对于特征的线性组合来拟合真实标记为正例的概率的对数几率

损失函数 交叉熵 -log（h（x））

解决过拟合问题的方法主要有两种： 
1. 减少特征数量，通过人工或者算法选择哪些特征有用保留，哪些特征没用删除，但会丢失信息。 
2. 正则化，保留特征，但减少特征对应参数的大小，让每个特征都对预测产生一点影响。

LR实现简单高效易解释，计算速度快，易并行，在大规模数据情况下非常适用，更适合于应对数值型和标称型数据，主要适合解决线性可分的问题，但容易欠拟合，大多数情况下需要手动进行特征工程，构建组合特征，分类精度不高。

LR直接对分类可能性进行建模，无需事先假设数据分布，这样就避免了假设分布不准确所带来的问题
LR能以概率的形式输出，而非知识0，1判定，对许多利用概率辅助决策的任务很有用
适用情景：LR是很多分类算法的基础组件，它的好处是输出值自然地落在0到1之间，并且有概率意义。
因为它本质上是一个线性的分类器，所以处理不好特征之间相关的情况。
虽然效果一般，却胜在模型清晰，背后的概率学经得住推敲。
它拟合出来的参数就代表了每一个特征(feature)对结果的影响。也是一个理解数据的好工具。

应用上： 
- CTR预估，推荐系统的learning to rank，各种分类场景 
- 某搜索引擎厂的广告CTR预估基线版是LR 
- 某电商搜索排序基线版是LR 
- 某新闻app排序基线版是LR

大规模工业实时数据，需要可解释性的金融数据，需要快速部署低耗时数据 
LR就是简单，可解释，速度快，消耗资源少，分布式性能好
'''
