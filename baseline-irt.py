# coding: utf-8
# 2021/5/2 @ liujiayu
import logging
import numpy as np
import pandas as pd
from EduCDM import EMIRT
# from EduData import get_data
import random

# #get_data("cdbd-a0910", "../../../data")

# # math
train_data = pd.read_csv("E:\dataset\objectmath2015\Math1\\train_data.csv")
# valid_data = pd.read_csv("../../../data/a0910/valid.csv")
test_data = pd.read_csv("E:\dataset\objectmath2015\Math1\\test_data.csv")

stu_num = max(max(train_data['user_id']), max(test_data['user_id']))
prob_num = max(max(train_data['item_id']), max(test_data['item_id']))

R = -1 * np.ones(shape=(stu_num, prob_num))
R[train_data['user_id']-1, train_data['item_id']-1] = train_data['score']

test_set = []
for i in range(len(test_data)):
    row = test_data.iloc[i]
    test_set.append({'user_id':int(row['user_id'])-1, 'item_id':int(row['item_id'])-1, 'score':row['score']})

logging.getLogger().setLevel(logging.INFO)

cdm = EMIRT(R, stu_num, prob_num, dim=1, skip_value=-1)  # IRT, dim > 1 is MIRT

cdm.train(lr=1e-4, epoch=5)
cdm.save("irt.params")

cdm.load("irt.params")
# rmse, mae, auc = cdm.eval(test_set)
# print("RMSE: %.6f, MAE: %.6f, AUC: %.6f" % (rmse, mae, auc))
rmse, mae = cdm.eval(test_set)
print("RMSE: %.6f, MAE: %.6f" % (rmse, mae))

# # ---incremental training
# # new_data = [{'user_id': 0, 'item_id': 2, 'score': 0.0}, {'user_id': 1, 'item_id': 1, 'score': 1.0}]
# # cdm.inc_train(new_data, lr=1e-3, epoch=2)
# #
# # # ---evaluate user's state
# # stu_rec = np.random.randint(-1, 2, size=prob_num)
# # dia_state = cdm.transform(stu_rec)
# # print("user's state is " + str(dia_state))
#
# # # dataprocess
# R = (np.loadtxt("E:\dataset\objectmath2015\Math1\\data.txt") == 1).astype(float)
# R = R[:, 0:15]
# new_R = np.zeros((R.shape[0]*R.shape[1], 3))  # 4209*15
# # user_id
# for i in range(R.shape[0]):
#     for j in range(R.shape[1]):
#         new_R[i*(R.shape[1])+j][0] = i+1
# # item_id
# for i in range(R.shape[0]):
#     for j in range(R.shape[1]):
#         new_R[i * (R.shape[1]) + j][1] = j + 1
# # score
# for i in range(R.shape[0]):
#     for j in range(R.shape[1]):
#         new_R[i * (R.shape[1]) + j][2] = R[i][j]
#
# #np.savetxt("E:\dataset\objectmath2015\Math1\\irt_data.csv", new_R, delimiter=',', fmt='%d')
# train = new_R[:2000,:]
# test = new_R[1600:2000,:]
# # valid = random.sample(list(new_R), 500*15)
# #np.savetxt("E:\dataset\objectmath2015\Math1\\valid_data.csv", valid, delimiter=',', fmt='%d')
# np.savetxt("E:\dataset\synthetic\\train_data.csv", train, delimiter=',', fmt='%d')
# np.savetxt("E:\dataset\objectmath2015\Math1\\test_data.csv", test, delimiter=',', fmt='%d')
# #

# cl19
# train_data = pd.read_csv("E:\mCDM-c\d-cdm\data\CL19\\train_data.csv")
# # valid_data = pd.read_csv("../../../data/a0910/valid.csv")
# test_data = pd.read_csv("E:\mCDM-c\d-cdm\data\CL19\\test_data.csv")
#
# stu_num = max(max(train_data['user_id']), max(test_data['user_id']))
# prob_num = max(max(train_data['item_id']), max(test_data['item_id']))
#
# R = -1 * np.ones(shape=(stu_num, prob_num))
# R[train_data['user_id']-1, train_data['item_id']-1] = train_data['score']
#
# test_set = []
# for i in range(len(test_data)):
#     row = test_data.iloc[i]
#     test_set.append({'user_id':int(row['user_id'])-1, 'item_id':int(row['item_id'])-1, 'score':row['score']})
#
# logging.getLogger().setLevel(logging.INFO)
#
# cdm = EMIRT(R, stu_num, prob_num, dim=1, skip_value=-1)  # IRT, dim > 1 is MIRT
#
# cdm.train(lr=1e-3, epoch=2)
# cdm.save("irt.params")
#
# cdm.load("irt.params")
# rmse, mae, auc = cdm.eval(test_set)
# print("RMSE: %.6f, MAE: %.6f, AUC: %.6f" % (rmse, mae, auc))


# # dataprocess
# R = (np.loadtxt("E:\mCDM-c\d-cdm\data\CL19\\CL19.txt") == 1).astype(float)
# new_R = np.zeros((R.shape[0]*R.shape[1], 3))  # 4209*15
# # user_id
# for i in range(R.shape[0]):
#     for j in range(R.shape[1]):
#         new_R[i*(R.shape[1])+j][0] = i+1
# # item_id
# for i in range(R.shape[0]):
#     for j in range(R.shape[1]):
#         new_R[i * (R.shape[1]) + j][1] = j + 1
# # score
# for i in range(R.shape[0]):
#     for j in range(R.shape[1]):
#         new_R[i * (R.shape[1]) + j][2] = R[i][j]
#
# #np.savetxt("E:\dataset\objectmath2015\Math1\\irt_data.csv", new_R, delimiter=',', fmt='%d')
#
# test = random.sample(list(new_R), 30*36)
# np.savetxt("E:\mCDM-c\d-cdm\data\CL19\\train_data.csv", new_R, delimiter=',', fmt='%d')
# np.savetxt("E:\mCDM-c\d-cdm\data\CL19\\test_data.csv", test, delimiter=',', fmt='%d')

# Synthetic
# train_data = pd.read_csv("E:\dataset\synthetic\\train_data.csv")
# # valid_data = pd.read_csv("../../../data/a0910/valid.csv")
# test_data = pd.read_csv("E:\dataset\synthetic\\test_data.csv")
#
# stu_num = max(max(train_data['user_id']), max(test_data['user_id']))
# prob_num = max(max(train_data['item_id']), max(test_data['item_id']))
#
# R = -1 * np.ones(shape=(stu_num, prob_num))
# R[train_data['user_id']-1, train_data['item_id']-1] = train_data['score']
#
# test_set = []
# for i in range(len(test_data)):
#     row = test_data.iloc[i]
#     test_set.append({'user_id':int(row['user_id'])-1, 'item_id':int(row['item_id'])-1, 'score':row['score']})
#
# logging.getLogger().setLevel(logging.INFO)
#
# cdm = EMIRT(R, stu_num, prob_num, dim=1, skip_value=-1)  # IRT, dim > 1 is MIRT
#
# cdm.train(lr=1e-3, epoch=2)
# cdm.save("irt.params")
#
# cdm.load("irt.params")
# rmse, mae, auc = cdm.eval(test_set)
# print("RMSE: %.6f, MAE: %.6f, AUC: %.6f" % (rmse, mae, auc))


# # dataprocess
# R = (np.loadtxt("E:\dataset\synthetic\\synthetic_R0_train.txt") == 1).astype(float)
# new_R = np.zeros((R.shape[0]*R.shape[1], 3))  # 4209*15
# # user_id
# for i in range(R.shape[0]):
#     for j in range(R.shape[1]):
#         new_R[i*(R.shape[1])+j][0] = i+1
# # item_id
# for i in range(R.shape[0]):
#     for j in range(R.shape[1]):
#         new_R[i * (R.shape[1]) + j][1] = j + 1
# # score
# for i in range(R.shape[0]):
#     for j in range(R.shape[1]):
#         new_R[i * (R.shape[1]) + j][2] = R[i][j]
#
# #np.savetxt("E:\dataset\objectmath2015\Math1\\irt_data.csv", new_R, delimiter=',', fmt='%d')
#
# test = random.sample(list(new_R), 500*50)
# np.savetxt("E:\dataset\synthetic\\train_data.csv", new_R, delimiter=',', fmt='%d')
# np.savetxt("E:\dataset\synthetic\\test_data.csv", test, delimiter=',', fmt='%d')