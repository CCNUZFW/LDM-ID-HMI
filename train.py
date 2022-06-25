import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score
# from saedata import TrainDataLoader, ValTestDataLoader
from data_loader_time import TrainDataLoader, ValTestDataLoader
from sklearn.metrics import mean_squared_error
from model import Net
import time

# can be changed according to config.txt
exer_n = 15
knowledge_n = 11
student_n = 4209
# can be changed according to command parameter
device = torch.device(('cuda:0') if torch.cuda.is_available() else 'cpu')
epoch_n = 5
torch.manual_seed(6)


def train():
    data_loader = TrainDataLoader()
    net = Net(student_n, exer_n, knowledge_n)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    #optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    print('training model...')

    loss_function = nn.NLLLoss()
    for epoch in range(epoch_n):
        #adjust_learning_rate(optimizer, epoch, lr=0.001)
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            q, input_stu, stu_theta, input_exer, labels = data_loader.next_batch()
            q, input_stu, stu_theta, input_exer, labels = q.to(device), input_stu.to(device), stu_theta.to(device), input_exer.to(device), labels.to(device)
            optimizer.zero_grad()
            output_1 = net.forward(q, input_stu, stu_theta, input_exer)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            grad_penalty = 0
            # labels = labels.view(32, -1)
            # label：shape是n，表示了n个向量对应的正确类别；
            # output: shape: (n, category),则表示每个类别预测的概率，比如向量（2，3，1）则表示类别0，1，2预测的概率分别为（2，3，1）

            # print('labels',labels)

            loss = loss_function(torch.log(output), labels.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_count % 9 == 1:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 10))
                running_loss = 0.0

        # validate and save current model every epoch
        rmse, auc = validate(net, epoch)
        save_snapshot(net, 'model/model_epoch' + str(epoch + 1))


def validate(model, epoch):
    data_loader = ValTestDataLoader('validation')
    since = time.time()
    net = Net(student_n, exer_n, knowledge_n)
    print('validating model...')
    data_loader.reset()
    # load model parameters
    net.load_state_dict(model.state_dict())
    net = net.to(device)
    net.eval()

    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        q, input_stu, stu_theta, input_exer, labels = data_loader.next_batch()
        q, input_stu, stu_theta, input_exer, labels = q.to(device), input_stu.to(device), stu_theta.to(device), input_exer.to(device), labels.to(device)
        output = net.forward(q, input_stu, stu_theta, input_exer)
        # print('labels',labels)
        # compute accuracy
        #print('out',output.shape)
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)

        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count

    # 根均方误差(RMSE)
    rmse2 = np.sqrt(mean_squared_error(label_all, pred_all))
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}s {:.0f}ms'.format(
        time_elapsed, time_elapsed * 1000))

    print('epoch= %d, accuracy= %f, rmse= %f,rmse2=%f, auc= %f' % (epoch+1, accuracy, rmse, rmse2, auc))
    with open('result/model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch+1, accuracy, rmse, auc))

    return rmse, auc


def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    if (len(sys.argv) != 3) or ((sys.argv[1] != 'cpu') and ('cuda:' not in sys.argv[1])) or (not sys.argv[2].isdigit()):
        print('command:\n\tpython train.py {device} {epoch}\nexample:\n\tpython train.py cuda:0 70')
        exit(1)
    else:
        device = torch.device(sys.argv[1])
        epoch_n = int(sys.argv[2])

    # global student_n, exer_n, knowledge_n, device
    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))
    train()
