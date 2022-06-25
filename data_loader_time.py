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
        self.ptr = 0

        data_file = 'E:\\neuralCDM1\\data\\Math2005\\data.txt'

        self.q = np.loadtxt('E:\\neuralCDM1\\data\\Math2005\\q.txt')
        self.feature = pd.read_csv('E:\\neuralCDM2\\data\\b.csv', header=None)
        self.sg = pd.read_csv('E:\\neuralCDM2\\data\\sg.csv', header=None,delimiter=',')
        self.sg = self.sg.to_numpy()
        self.sg = self.sg.T
        self.feature = self.feature.to_numpy()
        self.skills = pd.read_csv('E:\\neuralCDM1\\data\\est_skills.csv')
        self.theta = pd.read_csv('E:\\neuralCDM1\\data\\a.csv', header=None)
        self.theta = self.theta.to_numpy()
        self.skills = self.skills.to_numpy()
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
                input_stu_ids.append(count)
                input_exer_ids.append(j)
                # input_knowledge_embs.append(self.q[j])
                # input_exer_discrimination.append(self.feature[0][j])
                # input_exer_difficulty.append(self.feature[1][j])
                # input_exer_guess.append(self.feature[2][j])
                # input_exer_noslip.append(self.feature[3][j])
                input_exer.append(self.feature[j])
                input_dina.append(self.sg[j])
                input_stu.append(self.skills[count])
                #print(self.theta[count])
                input_stu_theta.append(self.theta[count])
                qm.append(self.q[j])

        self.ptr += self.batch_size
        # return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), \
        #        torch.LongTensor(input_exer_discrimination),torch.LongTensor(input_exer_difficulty),torch.LongTensor(input_exer_guess),torch.LongTensor(input_exer_noslip),torch.LongTensor(ys)
        # return torch.LongTensor(qm), torch.LongTensor(input_stu), torch.LongTensor(input_stu_theta),torch.FloatTensor(input_exer),torch.FloatTensor(ys)
        return torch.FloatTensor(qm), torch.FloatTensor(input_stu), torch.FloatTensor(
            input_stu_theta), torch.FloatTensor(
            input_exer), torch.FloatTensor(ys)
        #return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids),torch.LongTensor(input_knowledge_embs), torch.LongTensor(ys),


    def is_end(self):
        if self.ptr + self.batch_size > 3000:
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0


class ValTestDataLoader(object):
    def __init__(self, d_type='validation'):
        self.ptr = 3000
        self.data = []
        self.d_type = d_type
        data_file = 'E:\\neuralCDM1\\data\\Math2005\\data.txt'
        self.batch_size = 30
        self.feature = pd.read_csv('E:\\neuralCDM2\\data\\b.csv', header=None)
       # self.feature = self.feature.T
        self.feature = self.feature.to_numpy()
        self.sg = pd.read_csv('E:\\neuralCDM2\\data\\sg.csv', header=None,delimiter=',')
        self.sg = self.sg.to_numpy()
        self.sg = self.sg.T
        q = np.loadtxt('E:\\neuralCDM1\\data\\Math2005\\q.txt')
        self.skills = pd.read_csv('E:\\neuralCDM1\\data\\est_skills.csv')
        self.theta = pd.read_csv('E:\\neuralCDM1\\data\\a.csv',header=None)
        self.theta = self.theta.to_numpy()
        self.skills = self.skills.to_numpy()
        data = np.loadtxt(data_file)
        data = data[:, :15]
        q = q[:15, :]
        self.data = data
        self.q = q

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
                input_stu_ids.append(count)
                input_exer_ids.append(j)
                # input_knowledge_embs.append(self.q[j])
                # input_exer_discrimination.append(self.feature[0][j])
                # input_exer_difficulty.append(self.feature[1][j])
                # input_exer_guess.append(self.feature[2][j])
                # input_exer_noslip.append(self.feature[3][j])
                input_stu_theta.append(self.theta[count])
                input_exer.append(self.feature[j])
                input_dina.append(self.sg[j])
                input_stu.append(self.skills[count])
                qm.append(self.q[j])

        self.ptr += self.batch_size
        # return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(
        #     input_knowledge_embs),  \
        #        torch.LongTensor(input_exer_discrimination), torch.LongTensor(input_exer_difficulty), torch.LongTensor(
        #     input_exer_guess), torch.LongTensor(input_exer_noslip),torch.LongTensor(ys)
        # return torch.LongTensor(qm), torch.LongTensor(input_stu), torch.LongTensor(input_stu_theta),torch.FloatTensor(input_exer),torch.FloatTensor(ys)
        return torch.FloatTensor(qm), torch.FloatTensor(input_stu), torch.FloatTensor(
            input_stu_theta), torch.FloatTensor(
                    input_exer), torch.FloatTensor(ys)
        #return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids),torch.LongTensor(input_knowledge_embs), torch.LongTensor(ys)

    def is_end(self):
        if self.ptr >= 4000:
            return True
        else:
            return False

    def reset(self):
        self.ptr = 3000

# class TrainDataLoader(object):
#     '''
#     data loader for training
#     '''
#     def __init__(self):
#         self.batch_size = 36
#         self.ptr = 0
#
#         data_file = 'E:\\dataset\\CL19\\CL19.txt'
#         self.q = np.loadtxt('E:\\dataset\\CL19\\CL19Q_matrix.txt')
#
#         self.feature = pd.read_csv('E:\dataset\CL19\\features\\abcds.csv', header=None, delimiter=',')
#         self.feature = self.feature.to_numpy()
#         # self.feature = self.feature.T
#
#         self.sg = pd.read_csv('E:\dataset\CL19\\features\\sg.csv', header=None,delimiter=',')
#         self.sg = self.sg.to_numpy()
#
#         self.theta = pd.read_csv('E:\dataset\CL19\\features\\thetas.csv', header=None, delimiter=',')
#         self.skills = pd.read_csv('E:\dataset\CL19\\features\\est_skills.csv', header=None, delimiter=',')
#         self.theta = self.theta.to_numpy()
#         self.skills = self.skills.to_numpy()
#
#         data = np.loadtxt(data_file)
#         self.data = data
#
#     def next_batch(self):
#         if self.is_end():
#             return None, None, None, None
#         input_stu_ids, input_exer_ids, input_knowledge_embs, input_stu_theta,input_dina,qm,\
#         input_exer_discrimination, input_exer_difficulty, input_exer_guess, input_exer_noslip, \
#         ys = [], [], [], [], [], [], [], [], [],[],[]
#         input_exer = []
#         input_stu = []
#         for i in range(1):
#             for j in range(self.data.shape[1]):
#                 count = i + self.ptr
#                 ys.append(self.data[count][j])
#                 # input_stu_ids.append(self.skills[count])
#                 # input_stu_ids.append(count)
#                 # input_exer_ids.append(j)
#                 input_knowledge_embs.append(self.q[j])
#                 # input_exer_discrimination.append(self.feature[0][j])
#                 # input_exer_difficulty.append(self.feature[1][j])
#                 # input_exer_guess.append(self.feature[2][j])
#                 # input_exer_noslip.append(self.feature[3][j])
#                 input_exer.append(self.feature[j])
#                 # input_dina.append(self.sg[j])
#                 input_stu.append(self.skills[count])
#                 #print(self.theta[count])
#                 input_stu_theta.append(self.theta[count])
#                 qm.append(self.q[j])
#                 # input_stu_ids.append(count)
#                 # input_exer_ids.append(j)
#
#         self.ptr += 1
#         # return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), \
#         #        torch.FloatTensor(input_exer_discrimination),torch.FloatTensor(input_exer_difficulty),torch.FloatTensor(input_exer_guess),torch.FloatTensor(input_exer_noslip),torch.LongTensor(ys)
#         # return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids),torch.FloatTensor(qm), torch.FloatTensor(input_stu),torch.FloatTensor(input_stu_theta),torch.FloatTensor(input_exer),torch.FloatTensor(ys)
#
#         #return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids),torch.LongTensor(input_knowledge_embs), torch.LongTensor(ys),
#
#         return torch.FloatTensor(qm), torch.FloatTensor(input_stu), torch.FloatTensor(input_stu_theta), torch.FloatTensor(
#             input_exer), torch.FloatTensor(ys)
#
#
#     def is_end(self):
#         if self.ptr + self.batch_size > 60:
#             return True
#         else:
#             return False
#
#     def reset(self):
#         self.ptr = 0
#
#
# class ValTestDataLoader(object):
#     def __init__(self, d_type='validation'):
#         self.batch_size = 36
#         self.ptr = 60
#
#         data_file = 'E:\\dataset\\CL19\\CL19.txt'
#         self.q = np.loadtxt('E:\\dataset\\CL19\\CL19Q_matrix.txt')
#
#         self.feature = pd.read_csv('E:\dataset\CL19\\features\\abcds.csv', header=None, delimiter=',')
#         self.feature = self.feature.to_numpy()
#         # self.feature = self.feature.T
#
#         self.sg = pd.read_csv('E:\dataset\CL19\\features\\sg.csv', header=None, delimiter=',')
#         self.sg = self.sg.to_numpy()
#
#         self.theta = pd.read_csv('E:\dataset\CL19\\features\\thetas.csv', header=None, delimiter=',')
#         self.skills = pd.read_csv('E:\dataset\CL19\\features\\est_skills.csv', header=None, delimiter=',')
#         self.theta = self.theta.to_numpy()
#         self.skills = self.skills.to_numpy()
#
#         data = np.loadtxt(data_file)
#         self.data = data
#
#     def next_batch(self):
#         if self.is_end():
#             return None, None, None, None
#         input_stu_ids, input_exer_ids, input_knowledge_embs, input_stu_theta, input_dina, qm, \
#         input_exer_discrimination, input_exer_difficulty, input_exer_guess, input_exer_noslip, \
#         ys = [], [], [], [], [], [], [], [], [], [], []
#         input_exer = []
#         input_stu = []
#         for i in range(1):
#             for j in range(self.data.shape[1]):
#                 count = i + self.ptr
#                 ys.append(self.data[count][j])
#                 # input_stu_ids.append(self.skills[count])
#                 # input_stu_ids.append(count)
#                 # input_exer_ids.append(j)
#                 input_knowledge_embs.append(self.q[j])
#                 # input_exer_discrimination.append(self.feature[0][j])
#                 # input_exer_difficulty.append(self.feature[1][j])
#                 # input_exer_guess.append(self.feature[2][j])
#                 # input_exer_noslip.append(self.feature[3][j])
#                 input_exer.append(self.feature[j])
#                 # input_dina.append(self.sg[j])
#                 input_stu.append(self.skills[count])
#                 # print(self.theta[count])
#                 input_stu_theta.append(self.theta[count])
#                 qm.append(self.q[j])
#                 # input_stu_ids.append(count)
#                 # input_exer_ids.append(j)
#         self.ptr += 1
#         # return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(
#         #     input_knowledge_embs),  \
#         #        torch.FloatTensor(input_exer_discrimination), torch.FloatTensor(input_exer_difficulty), torch.FloatTensor(
#         #     input_exer_guess), torch.FloatTensor(input_exer_noslip),torch.LongTensor(ys)
#         # return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids),torch.FloatTensor(qm), torch.FloatTensor(input_stu), torch.FloatTensor(input_stu_theta), torch.FloatTensor(
#         #     input_exer), torch.FloatTensor(ys)
#
#         return torch.FloatTensor(qm), torch.FloatTensor(input_stu), torch.FloatTensor(input_stu_theta), torch.FloatTensor(
#             input_exer), torch.FloatTensor(ys)
#
#     def is_end(self):
#         if self.ptr >= 90:
#             return True
#         else:
#             return False
#
#     def reset(self):
#         self.ptr = 60
