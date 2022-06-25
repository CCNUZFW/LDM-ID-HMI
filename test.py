from __future__ import print_function, division, unicode_literals
import json
import torch
import numpy as np
from psy import Mirt2PL
from torch import autograd
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

# data_file = 'data/log_data.json'
# config_file = 'config.txt'
# with open(data_file, encoding='utf8') as i_f:
#     data = json.load(i_f)   # data[i]:{'user_id':i+1,'log_num','logs'{'exer_id','score','knoeledge_code'} * log_num }   user_id:1-4163
#
# knowledge_dim = 123

# input_stu_ids, input_exer_ids, input_knowedge_embs, ys = [], [], [], []
# r = np.zeros((32, 32))
# for count in range(32,64):
#     log = data[count]
#     knowledge_emb = [0.] * knowledge_dim
#     for knowledge_code in log['knowledge_code']:
#                 knowledge_emb[knowledge_code - 1] = 1.0
#     y = log['score']
#
#     r[count-32][count-32] = y
#
#     input_stu_ids.append(log['user_id'] - 1)
#     input_exer_ids.append(log['exer_id'] - 1)
#     input_knowedge_embs.append(knowledge_emb)
#     ys.append(y)
# s,t,l = Mirt2PL(scores=r, dim_size=2).em()
# discrimination = s[0]
# difficulty = t
# print(difficulty)
# print(discrimination)
# print(input_stu_ids)
# print(input_exer_ids)
# print(ys)

# log = data[1]
# logs = log['logs']
# logss = logs[1]
# print(logs)
# # [{'exer_id': 3, 'score': 0.0, 'knowledge_code': [1, 10]},
# # {'exer_id': 4, 'score': 1.0, 'knowledge_code': [1, 10]},
# # {'exer_id': 5, 'score': 0.0, 'knowledge_code': [1, 12]},
# print(logss)
# # {'exer_id': 4, 'score': 1.0, 'knowledge_code': [1, 10]}
#
# r = np.zeros((4163, 17746))
# for i in range(4163):
#     log = data[i]
#     log_num = log['log_num']
#     logs = log['logs']
#     for j in range(log_num):
#         record = logs[j]
#         if record['score'] == 1:
#             exer_id = record['exer_id']
#             r[i][exer_id-1] = 1
#
# np.savetxt('assistment_r.csv', r, delimiter = ',')

# data_file = 'E:\\neuralCDM\\data\\Math2005\\data.txt'
# batch_size = 32
# ptr = 0
# q = np.loadtxt('E:\\neuralCDM\\data\\Math2005\\q.txt')
# data = np.loadtxt(data_file)
# data = data[:, :15]
# q = q[:15, :]
#
# input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
#
# for i in range(2):
#     for j in range(data.shape[1]):
#             ys.append(data[i][j])
#             input_stu_ids.append(i)
#             input_exer_ids.append(j)
#             input_knowledge_embs.append(q[j])
#
# i += batch_size


class SelfAttention(nn.Module):
    """     scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.     """
    def __init__(self, d_hid, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)
        def forward(self, input_seq, lens):
            batch_size, seq_len, feature_dim = input_seq.size()
            input_seq = self.dropout(input_seq)
            scores = self.scorer(input_seq.contiguous().view(-1, feature_dim)).view(batch_size, seq_len)
            max_len = max(lens)
            for i, l in enumerate(lens):
                if l < max_len:
                    scores.data[i, l:] = -np.inf
                    scores = F.softmax(scores, dim=1)
                    context = scores.unsqueeze(2).expand_as(input_seq).mul(input_seq).sum(1)
                    return context # 既然命名为context就应该是整句的表示

