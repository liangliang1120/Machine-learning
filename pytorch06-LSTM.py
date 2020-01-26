# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 16:22:23 2020

@author: us
"""

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

torch.manual_seed(2019)
  
train_data = datasets.MNIST(
  	root='data/',
  	train=True,
  	transform=transforms.ToTensor(),
  	download=False,  # 第一次使用需要下载
  )
test_data = datasets.MNIST(
  	root='data/',
  	train=False
  )
  
dataloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=4)  # 为了配合GPU性能最好使用2^n
# 为了配合PyTorch模型的输入格式(N, C, W, H)，这里调整数据格式
x_test = test_data.data.type(torch.FloatTensor)[:200] / 255
y_test = test_data.targets.numpy()[:200]


# 定义网络结构
import torch.nn as nn
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
  
        self.rnn = nn.LSTM(  # 这里使用LSTM（长短时记忆单元）
  			input_size=28,  # 图片的行数据
  			hidden_size=64,  # 隐层单元数目
  			num_layers=1,  # 这样的层数目
  			batch_first=True,  # batch_size是否为第一个维度如(batch_size, time_step, input_size)
  		)
        self.output = nn.Linear(64, 10)  # 这里使用全连接输出
  
    def forward(self, x):
        """
  		输入数据x格式为(batch_size, time_step, input_size)
  		输出r_output数据格式为(batch_size, time_step, output_size)
  		h_n即hidden state1格式为(n_layers, batch, hidden_size) 
  		h_c即hidden state2格式为(n_layers, batch, hidden_size) 
  		"""
        r_output, (h_n, h_c) = self.rnn(x, None)  # None表示hidden state使用全0
        output = self.output(r_output[:, -1, :])  # 取最后一个时间点数据
        return output
  
rnn = RNN()
print(rnn)


# 进行训练
import torch.optim as optim
import numpy as np
optimizer = optim.Adam(rnn.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()  # 分类一般使用交叉熵损失函数
  
for epoch in range(50):
	for step, (b_x, b_y) in enumerate(dataloader):
  		b_x = b_x.view(-1, 28, 28)
  		output = rnn(b_x)
  		loss = loss_func(output, b_y)
  		
  		
  		optimizer.zero_grad()
  		loss.backward() 
  		optimizer.step() 
  
  		if step % 10 == 0:
  			test_output = rnn(x_test)
  			pred_y = torch.max(test_output, 1)[1].data.numpy()
  			accuracy = np.sum(pred_y == y_test) / y_test.size
  			print('Epoch: {}'.format(epoch), 'train loss: {:.4f}'.format(loss.data.numpy()), 'test accuracy: {:.2f}'.format(accuracy))
  
# 原文链接：https://blog.csdn.net/zhouchen1998/article/details/89480618

























