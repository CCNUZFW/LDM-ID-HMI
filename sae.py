import torch
import torchvision
import torch.utils.data as Data
import torch.nn as nn
from saedata import saedata1
import numpy as np
import os
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import time

starttime = time.time()

torch.manual_seed(1)  # 为了使用同样的随机初始化种子以形成相同的随机效果

EPOCH = 50
LR = 0.1
data_loader = saedata1()

# # 题目
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder  =  nn.Sequential(
            nn.Linear(60, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            # nn.Linear(64, 32),
            # nn.Tanh(),
            # nn.Linear(32, 16),
            # nn.Tanh(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            # nn.Linear(32, 64),
            # nn.Tanh(),
            # nn.Linear(16, 30),
            # nn.Tanh(),
            nn.Linear(128, 60),
            # nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# # 学生
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        self.encoder  =  nn.Sequential(
            nn.Linear(104, 128),
            nn.Tanh(),
            # nn.Linear(256, 128),
            # nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            # nn.Linear(32, 16),
            # nn.Tanh(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            # nn.Linear(128, 256),
            # nn.Tanh(),
            # nn.Linear(16, 32),
            # nn.Tanh(),
            nn.Linear(128, 104),
            nn.Sigmoid()
        )
    def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded,decoded

Coder = AutoEncoder()
# print(Coder)

optimizer = torch.optim.Adam(Coder.parameters(), lr=LR)
loss_func = nn.MSELoss()
for epoch in range(120):
    data_loader.reset()
    running_loss = 0.0
    batch_count = 0
    while not data_loader.is_end():
        batch_count += 1
        a,b,c,d = data_loader.next_batch()
        b_label = d
        # x = torch.cat((a,b),dim=1)   # 30,104
        x = c  # 30 ,60

        b_x = x.view(-1, 60)  #(batch_size , 16)

        b_y = x.view(-1, 60)

    encoded, decoded = Coder(b_x)
   # print(decoded.shape)
    loss = loss_func(decoded, b_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if batch_count % 10 == 0:
        print('Epoch :', epoch, '|', 'train_loss:%.4f' % loss.data)
#torch.save(Coder, 'AutoEncoder_student.pkl')
print('________________________________________')
print('finish training')

#
# path = 'E:\\neuralCDM2\\ori_net.pkl'
# ori_net = AutoEncoder()
# state_dict = ori_net.state_dict()
# torch.save(state_dict, path)
# print(list(ori_net.parameters()))

# for name in Coder.state_dict():
#   print(name)

'''
sae结构：
encoder.0.weight
encoder.0.bias
encoder.2.weight
encoder.2.bias
decoder.0.weight
decoder.0.bias
decoder.2.weight
decoder.2.bias
'''
# weight_exer = Coder.state_dict()['decoder.2.weight']
# bias_exer = Coder.state_dict()['decoder.2.bias']
# weight_exer.numpy()
# bias_exer.numpy()
# np.savetxt("weight_exer.csv", weight_exer)
# np.savetxt("bias_exer.csv",bias_exer)

# 保存最后一层模型参数
# weight_stu = Coder.state_dict()['decoder.2.weight']
# bias_stu = Coder.state_dict()['decoder.2.bias']
# weight_stu.numpy()
# bias_stu.numpy()
# np.savetxt("weight_stu.csv", weight_stu)
# np.savetxt("bias_stu.csv", bias_stu)

# s_w = np.loadtxt("weight_stu.csv")
# s_b = np.loadtxt("bias_stu.csv")
# e_w = np.loadtxt("weight_exer.csv")
# e_b = np.loadtxt("bias_exer.csv")
# print(s_w.shape, s_b.shape, e_w.shape, e_b.shape)



# example
torch.save(Coder, 'ckp/sae_stu.pth')
# sea_exer = Coder.load_state_dict(torch.load('ckp/sae_exer.pth'))