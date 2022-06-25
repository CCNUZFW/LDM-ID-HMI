import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')

class eAutoEncoder(nn.Module):
    def __init__(self):
        super(eAutoEncoder, self).__init__()
        self.encoder  =  nn.Sequential(
            nn.Linear(15, 64),
            nn.Tanh(),
            # nn.Linear(128, 64),
            # nn.Tanh(),
            # nn.Linear(64, 32),
            # nn.Tanh(),
            # nn.Linear(32, 16),
            # nn.Tanh(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.Tanh(),
            # nn.Linear(64, 128),
            # nn.Tanh(),
            # nn.Linear(32, 64),
            # nn.Tanh(),
            # nn.Linear(16, 30),
            # nn.Tanh(),
            nn.Linear(64, 15),
            # nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# # 学生
class sAutoEncoder(nn.Module):
    def __init__(self):
        super(sAutoEncoder,self).__init__()
        self.encoder  =  nn.Sequential(
            nn.Linear(12, 128),
            nn.Tanh(),
            # nn.Linear(256, 128),
            # nn.Tanh(),
            nn.Linear(128, 64),
            # nn.Tanh(),
            # # nn.Linear(32, 16),
            # # nn.Tanh(),
            # nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            # nn.Linear(32, 64),
            # nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            # nn.Linear(128, 256),
            # nn.Tanh(),
            # nn.Linear(16, 32),
            # nn.Tanh(),
            nn.Linear(128, 12),
            nn.Sigmoid()
        )

    def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded


class selfattention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels , kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels , kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        # input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        # input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  # tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        # np.savetxt('E:\\atten.csv', attn_matrix[0].cpu().detach().numpy(),delimiter='\t')
        out = out.view(*input.shape)
        out = self.gamma * out + input
        out = out.view(30, 1, -1)
        return out

class Net(nn.Module):
    '''
    NeuralCDM
    '''

    def __init__(self, student_n, exer_n, knowledge_n):
        self.knowledge_dim = 30
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = 30
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2, self.prednet_len3 = 256, 128, 64  # changeable
        self.n_layer = 2
        self.dropout = 0.2

        super(Net, self).__init__()

        self.attention = selfattention(1)
        self.ssae = sAutoEncoder()
        self.esae = eAutoEncoder()

        """
        # network structure
        self.student_emb = nn.Embedding(self.emb_num, 11)

        self.slip = nn.Embedding(self.exer_n, 11)
        self.guess = nn.Embedding(self.exer_n, 11)

        self.slip1 = nn.Embedding(self.exer_n, 11)
        self.guess1 = nn.Embedding(self.exer_n, 11)

        self.e_discrimination = nn.Embedding(self.exer_n, 11)
        self.k_difficulty = nn.Embedding(self.exer_n, 11)

        self.e_difficulty = nn.Embedding(self.exer_n, 11)
        self.k_discrimination = nn.Embedding(self.exer_n, 11)

        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.3)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.3)
        # self.prednet_full3 = nn.Linear(self.prednet_len2, 1)
        self.RNN = nn.RNNCell(30, 16)

        self.FULL1 = nn.Linear(self.prednet_len1, 512)
        self.drop_3 = nn.Dropout(p=0.5)
        self.FULL2 = nn.Linear(512, self.prednet_len1)
        self.drop_4 = nn.Dropout(p=0.5)
        self.lstm = nn.LSTM(192, self.prednet_len2, self.n_layer, dropout=0.3,
                            batch_first=True, bidirectional=True)
        """

        self.conv1 = nn.Conv1d(
            in_channels=96,
            out_channels=30,
            kernel_size=3,
            padding=1,
            bias=True
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = torch.nn.MaxPool1d(kernel_size=3,
                                        stride=3,
                                        padding=0,
                                        dilation=1,
                                        return_indices=False,
                                        ceil_mode=True)

        self.conv2 = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
            bias=True
        )
        self.relu2 = nn.Softmax
        self.pool2 = torch.nn.MaxPool1d(kernel_size=2,
                                        stride=2,
                                        padding=0,
                                        dilation=1,
                                        return_indices=False,
                                        ceil_mode=True)
        self.prednet_pro = nn.Linear(104, 11)
        self.prednet_full3 = nn.Linear(104, 128)
        self.prednet_full4 = nn.Linear(30, 1)

        self.prednet_full7 = nn.Linear(96, 64)
        self.prednet_full8 = nn.Linear(64, 32)
        self.prednet_full9 = nn.Linear(64, 1)
        self.drop = nn.Dropout(p=self.dropout)
        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, q, stu, theta, exer):
        """
        # e_discrimination = torch.sigmoid(self.e_discrimination(eid)) * 10
        # stu_emb = torch.sigmoid(self.student_emb(sid))
        # k_difficulty = torch.sigmoid(self.k_difficulty(eid))
        # input_x = e_discrimination * (stu_emb - k_difficulty) * q
        # print(input_x.shape)

        # sid = torch.sigmoid(self.student_emb(sid))
        # eid = torch.sigmoid(self.e_discrimination(eid))
        # input_stu = nn.functional.normalize(input_stu, p=2, dim=1)
        # input_exer= nn.functional.normalize(input_exer, p=2, dim=1)
        """
        input_exer = torch.cat((q, exer), dim=1)
        input_stu = torch.cat((stu, theta), dim=1)
        # print(input_exer.shape, input_stu.shape)

        input_stu, x = self.ssae.forward(input_stu)
        input_exer, y = self.esae.forward(input_exer)

        # input_stu = torch.from_numpy(np.array(input_stu))
        # input_exer = torch.from_numpy(np.array(input_exer))

        # input_exer = nn.functional.normalize(input_exer, p=2, dim=1)
        # input_stu = nn.functional.normalize(input_stu, p=2, dim=1)
        #  input_stu = torch.sigmoid(self.prednet_full3(input_stu))
        #  #input_stu = torch.sigmoid(self.prednet_full6(input_stu))  #128

        #  input_exer = torch.sigmoid(self.prednet_full4(input_exer))  #64
        #  input_exer = torch.sigmoid(self.prednet_full8(input_exer))

        input_x = torch.cat((input_stu, input_exer), dim=1)  # (30,96)
        # np.savetxt('E:\\FEA3.csv', input_x.cpu().detach().numpy(), delimiter=',')
        input_x = input_x.view(30, 1, -1, 1)
        #
        input_x = self.attention(input_x)
        # print(input_x.shape)
        output = input_x
        input_x = input_x.view(30, -1, 1)
        #

        output = self.conv1(input_x)
        output = self.relu1(output)
        output = self.pool1(output)
        output = output.view(30, -1)
        output = torch.sigmoid(self.prednet_full4(output))

        # output = torch.sigmoid(self.prednet_full7(input_x))
        # output = torch.sigmoid(self.drop(output))
        # # output = torch.sigmoid(self.prednet_full8(output))
        # output = torch.sigmoid(self.prednet_full9(output))
        # output = output.view(30, 1)

        # output_1 = output.data.cpu().numpy()
        # np.savetxt('E:\dataset\objectmath2015\Math1\\vis\\output.txt', output_1)
        return output

    # def forward(self, stu, exer):
    #     '''
    #     :param stu_id: LongTensor
    #     :param exer_id: LongTensor
    #     :param kn_emb: FloatTensor, the knowledge relevancy vectors
    #     :return: FloatTensor, the probabilities of answering correctly
    #     '''
    #     # before prednet
    #     # stu_emb = torch.sigmoid(self.student_emb(stu_id))
    #     # #
    #     # slip = torch.sigmoid(self.slip(exer_id)) * 2
    #     # guess = torch.sigmoid(self.guess(exer_id))
    #     #
    #     # slip1 = torch.sigmoid(self.slip1(exer_id))
    #     # guess1 = torch.sigmoid(self.guess1(exer_id))
    #     #
    #     # e_discrimination = torch.sigmoid(self.e_discrimination(exer_id))
    #     # k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
    #     #
    #     # e_difficulty = torch.sigmoid(self.e_difficulty(exer_id)) * 10
    #     # k_discrimination = torch.sigmoid(self.k_discrimination(exer_id))
    #
    #     # 学生能力离散化
    #     # stu_emb = (stu_emb>0.5).float()
    #
    #     # prednet
    #     # alpha = stu_emb.detach().cuda().data.cpu().numpy()
    #     # k = kn_emb.detach().cuda().data.cpu().numpy()
    #     # slip = slip.detach().cuda().data.cpu().numpy() * 0.5
    #     # guess = guess.detach().cuda().data.cpu().numpy() * 0.5
    #
    #     # NIDA
    #     # input_x = pow(pow(((np.ones((slip.shape[0], slip.shape[1])))-slip), alpha) * \
    #     #           pow(guess,(np.ones((slip.shape[0], slip.shape[1]))-alpha)), k)
    #
    #     # embedding的结果大小都是0.5左右，无明显差异
    #     # print('guess',guess.shape)
    #     # # print('slip',slip)
    #     # # print('e_discrimination',e_discrimination)
    #     # # print('k_difficulty',k_difficulty)
    #     # print('stu_emb',stu_emb.shape)
    #     # print(kn_emb.shape)
    #     #
    #     # input_x_1 = (stu_emb - slip + guess + e_difficulty) * kn_emb * e_discrimination
    #     # print(input_x_1.shape)
    #     # input_x_2 = exer_discrimination * (stu_emb - exer_difficulty) * kn_emb
    #     # input_x_3 = exer_difficulty * exer_discrimination * stu_emb * kn_emb
    #     # input_x_4 = (stu_emb - (1-exer_noslip) +exer_guess) * kn_emb
    #     #
    #     # # print('input_x',input_x)
    #     # input_x_1 = self.drop_1(torch.sigmoid(self.prednet_full1(input_x_1)))
    #     # input_x_1 = self.drop_2(torch.sigmoid(self.prednet_full2(input_x_1)))
    #     #
    #     # input_x_2 = self.drop_1(torch.sigmoid(self.prednet_full1(input_x_2)))
    #     # input_x_2 = self.drop_2(torch.sigmoid(self.prednet_full2(input_x_2)))
    #     #
    #     # input_x_3 = self.drop_1(torch.sigmoid(self.prednet_full1(input_x_3)))
    #     # input_x_3 = self.drop_2(torch.sigmoid(self.prednet_full2(input_x_3)))
    #     #
    #     # input_x_4 = self.drop_1(torch.sigmoid(self.prednet_full1(input_x_4)))
    #     # input_x_4 = self.drop_2(torch.sigmoid(self.prednet_full2(input_x_4)))
    #
    #     # cat
    #     # input_x = torch.cat((input_x_1, input_x_2), 1)  # (32,256)
    #     # input_x = nn.functional.normalize(input_x, p=2, dim=1)
    #
    #     # input_x = self.drop_3(torch.sigmoid(self.FULL1(input_x)))
    #     # input_x = self.drop_4(torch.sigmoid(self.FULL2(input_x)))
    #     # l = len(input_x)
    #     # input_x = input_x.view(l, 1, -1).float().to(device='cuda:0')
    #
    #     # print(input_x.shape)
    #     # print(input_x.shape)
    #
    #     # # RNN
    #     # input_stu_ids = list(map(list, zip(*stu_id)))
    #     # b = np.dot(kn_emb, input_stu_ids)
    #     # b = b[:, 0]
    #     # b = torch.from_numpy(b).to(device='cuda:0')
    #     # input_x = (b - (1-exer_noslip) + exer_guess - exer_difficulty) * exer_discrimination
    #     # # input_x = (b - (1 - exer_noslip) + exer_guess )
    #     # # out = self.RNN(input_x)
    #     # input_x = input_x.float().to(device='cuda:0')
    #     # output = torch.sigmoid(self.prednet_full3(input_x))
    #     # output = torch.sigmoid(self.prednet_full4(output))
    #     # output = output.reshape(30,-1)
    #
    #     # # CNN
    #    #  output = self.conv1(input_x)
    #    # # output = torch.sigmoid(output)
    #    #  output = self.pool1(output)
    #    #
    #    #  output = self.conv2(output)
    #    #  output = torch.sigmoid(output)
    #    #  output = self.pool2(output)
    #    #  output.reshape(l, -1)
    #    #  output = torch.sigmoid(self.prednet_full3(output))  # (32,1)
    #    #  output = output[:, 0]
    #   #  print(output)
    #
    #
    #     # LSTM
    #     # input_x = input_x.view(l, 1, -1).float().to(device='cuda:0')
    #     # out, (h_n, c_n) = self.lstm(input_x)
    #     # # 此时可以从out中获得最终输出的状态h
    #     # x = out[:, -1, :]
    #     # x = x.reshape(l, 1, -1)
    #     # out = out[:, :, :self.prednet_len2] + out[:, :, self.prednet_len2:]
    #     # # print(x.shape)
    #
    #     #out.reshape(l, -1)
    #    #  x = h_n[-1, :, :]
    #    #  # x = torch.unsqueeze(x,1)
    #    # # output, attn_weights = self.attention(out)
    #    #  output = torch.sigmoid(self.prednet_full3(out))
    #    # output = output[:, 0]
    #    #  print(output)
    #    #  return output

    # 为满足单调性假设，限制权值为正
    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        slip = torch.sigmoid(self.slip(exer_id))
        guess = torch.sigmoid(self.guess(exer_id))  # * 10
        return slip.data, guess.data


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
