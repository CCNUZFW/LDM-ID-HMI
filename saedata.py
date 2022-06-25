import json
import torch
import numpy as np
import pandas as pd

class TrainDataLoader(object):
    '''
    data loader for training
    '''
    def __init__(self):
        self.batch_size = 30
        self.ptr = 1000

        data_file = 'E:\\neuralCDM1\\data\\Math2005\\data.txt'

        self.q = np.loadtxt('E:\\neuralCDM1\\data\\Math2005\\q.txt')
        self.feature = np.load('E:\\neuralCDM3\\data\\onehot\\feature_onehot.npy')
        self.sg = pd.read_csv('E:\\neuralCDM2\\data\\sg.csv', header=None, delimiter=',')
        self.sg = self.sg.to_numpy()
        self.sg = self.sg.T
        # self.feature = self.feature.to_numpy()
        self.skills = np.load('E:\\neuralCDM3\\data\\onehot\\skills_onehot.npy')
        self.theta = np.load('E:\\neuralCDM3\\data\\onehot\\theta_onehot.npy')
        # self.theta = self.theta.to_numpy()
        # self.skills = self.skills.to_numpy()
        data = np.loadtxt(data_file)
        data = data[:, :15]
        self.q = self.q[:15, :]
        self.data = data

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowledge_embs, input_stu_theta,input_dina,qm,\
        input_exer_discrimination, input_exer_difficulty, input_exer_guess, input_exer_noslip, \
        ys = [], [], [], [], [], [], [], [], [],[],[]
        input_exer = []
        input_stu = []
        for i in range(2):
            for j in range(self.data.shape[1]):
                count = i + self.ptr
                ys.append(self.data[count][j])
                # input_stu_ids.append(self.skills[count])
                # input_stu_ids.append(count)
                # input_exer_ids.append(j)
                # input_knowledge_embs.append(self.q[j])
                # input_exer_discrimination.append(self.feature[0][j])
                # input_exer_difficulty.append(self.feature[1][j])
                # input_exer_guess.append(self.feature[2][j])
                # input_exer_noslip.append(self.feature[3][j])
                input_exer.append(self.feature[j])
                # input_dina.append(self.sg[j])
                input_stu.append(self.skills[count])
                #print(self.theta[count])
                input_stu_theta.append(self.theta[count])
                qm.append(self.q[j])

        self.ptr += self.batch_size
        # return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), \
        #        torch.LongTensor(input_exer_discrimination),torch.LongTensor(input_exer_difficulty),torch.LongTensor(input_exer_guess),torch.LongTensor(input_exer_noslip),torch.LongTensor(ys)
        return torch.FloatTensor(qm), torch.FloatTensor(input_stu),torch.FloatTensor(input_stu_theta),torch.FloatTensor(input_exer),torch.FloatTensor(ys)

        #return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids),torch.LongTensor(input_knowledge_embs), torch.LongTensor(ys),


    def is_end(self):
        if self.ptr + self.batch_size > 4000:
            return True
        else:
            return False

    def reset(self):
        self.ptr = 1000


class ValTestDataLoader(object):
    def __init__(self, d_type='validation'):
        self.ptr = 500
        self.data = []
        self.d_type = d_type
        data_file = 'E:\\neuralCDM1\\data\\Math2005\\data.txt'
        self.batch_size = 30
        self.q = np.loadtxt('E:\\neuralCDM1\\data\\Math2005\\q.txt')
        self.feature = np.load('E:\\neuralCDM3\\data\\onehot\\feature_onehot.npy')
        self.sg = pd.read_csv('E:\\neuralCDM2\\data\\sg.csv', header=None, delimiter=',')
        self.sg = self.sg.to_numpy()
        self.sg = self.sg.T
        # self.feature = self.feature.to_numpy()
        self.skills = np.load('E:\\neuralCDM3\\data\\onehot\\skills_onehot.npy')
        self.theta = np.load('E:\\neuralCDM3\\data\\onehot\\theta_onehot.npy')
        data = np.loadtxt(data_file)
        data = data[:, :15]
        # q = q[:15, :]
        self.data = data
        # self.q = q

    def next_batch(self):
        if self.is_end():
            return None, None, None, None

        input_stu_ids, input_exer_ids, input_knowledge_embs,input_stu_theta, input_dina,qm,\
        input_exer_discrimination, input_exer_difficulty, input_exer_guess, input_exer_noslip, \
        ys = [], [], [], [], [], [], [], [], [], [], []
        input_exer = []
        input_stu = []
        for i in range(2):
            for j in range(self.data.shape[1]):
                count = i + self.ptr
                ys.append(self.data[count][j])
                # input_stu_ids.append(self.skills[count])
                # input_stu_ids.append(count)
                # input_exer_ids.append(j)
                # input_knowledge_embs.append(self.q[j])
                # input_exer_discrimination.append(self.feature[0][j])
                # input_exer_difficulty.append(self.feature[1][j])
                # input_exer_guess.append(self.feature[2][j])
                # input_exer_noslip.append(self.feature[3][j])
                input_exer.append(self.feature[j])
                # input_dina.append(self.sg[j])
                input_stu.append(self.skills[count])
                # print(self.theta[count])
                input_stu_theta.append(self.theta[count])
                qm.append(self.q[j])
        self.ptr += self.batch_size
        # return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(
        #     input_knowledge_embs),  \
        #        torch.LongTensor(input_exer_discrimination), torch.LongTensor(input_exer_difficulty), torch.LongTensor(
        #     input_exer_guess), torch.LongTensor(input_exer_noslip),torch.LongTensor(ys)

        return torch.FloatTensor(qm), torch.FloatTensor(input_stu), torch.FloatTensor(input_stu_theta), torch.FloatTensor(
            input_exer), torch.FloatTensor(ys)

    def is_end(self):
        if self.ptr >= 1500:
            return True
        else:
            return False

    def reset(self):
        self.ptr = 500
