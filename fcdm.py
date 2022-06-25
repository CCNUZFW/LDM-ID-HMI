import numpy as np
from psy import Grm
from psy import Irt2PL
from psy import EmDina, MlDina
from psy.utils import r4beta
from psy import McmcDina
import pandas as pd
import torch

class TrainDataLoader(object):
    '''
    data loader for training
    '''
    def __init__(self):
        self.batch_size = 32
        self.ptr = 0
        self.data = []

        data_file = 'E:\\neuralCDM\\data\\Math2005\\data.txt'
        config_file = 'config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
           # self.data.sort(key=lambda x: x["user_id"])
        with open(config_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
        self.knowledge_dim = int(knowledge_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, ys = [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            y = log['score']
            input_stu_ids.append(log['user_id'] - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            input_knowedge_embs.append(knowledge_emb)
            ys.append(y)

        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.LongTensor(ys)

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0


def file_read(file_road):
    """
    数据读入
    """
    df = np.loadtxt(file_road)
    return df


q = file_read('E:\\neuralCDM1\\data\\Math2005\\q.txt')
score = file_read('E:\\neuralCDM1\\data\\Math2005\\data.txt')
# 取客观题部分
q = q[:15, :]
score = score[:, :15]
theta = pd.read_csv('E:\\neuralCDM2\\data\\sg.csv',header=None,delimiter=',')

theta=theta.to_numpy()
theta=theta.T
print(theta[14])
# irt
# discrimination, difficulty, theta,t = Irt2PL(scores=score).em()
# print(theta.shape)
#
# # dina
# attrs = q.T   # 知识点-试题（11,15）
# skills = np.random.binomial(1, 0.6, (4209, 11))
# g = r4beta(1, 2, 0, 0.6, (1, 15))
# no_s = r4beta(2, 1, 0.4, 1, (1, 15))
# temp = McmcDina(attrs=attrs)
# yita = temp.get_yita(skills)
# p_val = temp.get_p(yita, guess=g, no_slip=no_s)
# em_dina = McmcDina(attrs=attrs, score=score, max_iter=100, burn=50)
# est_skills, est_no_slip, est_guess = em_dina.mcmc()
# np.savetxt('E:\\neuralCDM\\data\\est_skills.csv', est_skills, delimiter=',')
# # est_skills.shape(4209, 11)

# print('discrimination', discrimination)
# print('difficulty', difficulty)
# print('est_skills', est_skills)
# print('est_guess', est_guess)
# print('est_no_slip', est_no_slip)


# feature = file_read('E:\\neuralCDM1\\data\\config.csv')

'''
数据输入
'''
# data_file = 'E:\\neuralCDM1\\data\\Math2005\\data.txt'
# batch_size = 32
# ptr = 0
# q = np.loadtxt('E:\\neuralCDM1\\data\\Math2005\\q.txt')
# skills = pd.read_csv('E:\\neuralCDM1\\data\\est_skills.csv')
# data = np.loadtxt(data_file)
# data = data[:, :15]
# q = q[:15, :]
#
# input_stu_ids, input_exer_ids, input_knowledge_embs,\
# input_exer_discrimination,input_exer_difficulty,input_exer_guess,input_exer_noslip, \
# ys = [], [], [], [], [], [], [], []
# skills = skills.to_numpy()
# feature = feature.T
# for i in range(2):
#     for j in range(data.shape[1]):
#         ys.append(data[i][j])
#         input_stu_ids.append(skills[i])
#         input_exer_ids.append(j)
#         input_knowledge_embs.append(q[j])
#         input_exer_discrimination.append(feature[j])
#         # input_exer_difficulty.append(feature[1][j])
#         # input_exer_guess.append(feature[2][j])
#         # input_exer_noslip.append(feature[3][j])
# i += batch_size
# #
#
# # input_stu_ids = list(map(list,zip(*input_stu_ids)))
# # b = np.dot(input_knowledge_embs , input_stu_ids)
# # b = b[:, 0]
# # print(b)
# print(input_stu_ids)
# exer_features = input_exer_noslip

