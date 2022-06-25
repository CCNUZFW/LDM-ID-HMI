'''
Author       : maywzh
Date         : 2021-04-05 22:28:39
LastEditTime : 2021-04-09 07:55:25
LastEditors  : maywzh
Description  :
FilePath     : /ji_coursenotes/2021spring/CCNUMaster/exp/chapter3/gkvmn/draw.py
symbol_custom_string_obkoro1:
Copyright (c) 2017 maywzh.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
# import pandas as pd
import re
file_name_lstm = 'E:\\results\\in4.txt'
file_name_rnn = 'E:\\results\\lstm.txt'
file_name_ori = 'E:\\results\\ori-train.txt'
#file_name_ls = 'E:\\neuralCDM\\lstm-train.txt'
ls = []
loss_lstm = []
loss_rnn = []
loss_l = []
loss_ori = []
auc_lstm = []
auc_rnn = []
auc_ori = []
acc = []
vauc = []
vacc = []
# with open(file_name_lstm) as f:
#     ls = f.readlines()
# cnt = 0
# for l in ls:
#     matloss = re.search('loss: ', l)
#     # matvalid = re.search('auc : ', l)
#     if matloss:
#         start = matloss.span()[1]
#         #print(l[start:start+7], l[start+15:start+22], l[start+35:start+42])
#         loss_lstm.append(float(l[start:start+6]))
#         cnt += 1
#
# with open(file_name_rnn) as f:
#     ls = f.readlines()
# cnt = 0
# for l in ls:
#     matloss = re.search('loss: ', l)
#     # matvalid = re.search('auc : ', l)
#     if matloss:
#         start = matloss.span()[1]
#         #print(l[start:start+7], l[start+15:start+22], l[start+35:start+42])
#         loss_rnn.append(float(l[start:start+6]))
#         cnt += 1
#
# with open(file_name_ori) as f:
#     ls = f.readlines()
# cnt = 0
# for l in ls:
#     matloss = re.search('loss: ', l)
#     # matvalid = re.search('auc : ', l)
#     if matloss:
#         start = matloss.span()[1]
#         #print(l[start:start+7], l[start+15:start+22], l[start+35:start+42])
#         loss_ori.append(float(l[start:start+6]))
#         cnt += 1


with open(file_name_lstm) as f:
    ls = f.readlines()
cnt = 0
for l in ls:
    matloss = re.search('auc=', l)
    if matloss:
        start = matloss.span()[1]
        auc_lstm.append(float(l[start:start+8]))
        cnt += 1

with open(file_name_rnn) as f:
    ls = f.readlines()
cnt = 0
for l in ls:
    matloss = re.search('auc=', l)
    # matvalid = re.search('auc : ', l)
    if matloss:
        start = matloss.span()[1]
        auc_rnn.append(float(l[start:start+8]))
        cnt += 1

with open(file_name_ori) as f:
    ls = f.readlines()
cnt = 0
for l in ls:
    matloss = re.search('auc=', l)
    # matvalid = re.search('auc : ', l)
    if matloss:
        start = matloss.span()[1]
        auc_ori.append(float(l[start:start+8]))
        cnt += 1

print(auc_rnn)
plt.figure(dpi=300)
# plt.plot(range(145), loss_lstm, label="cnn")
# plt.plot(range(145), loss_rnn, label="lstm")
# plt.plot(range(145), loss_ori, label="base")

plt.plot(range(5), auc_lstm, label="cnn")
plt.plot(range(5), auc_rnn, label="lstm")
plt.plot(range(5), auc_ori, label="base")
plt.title("training auc")
plt.xlabel("epoch")
plt.ylabel("")
plt.legend()
plt.show()
