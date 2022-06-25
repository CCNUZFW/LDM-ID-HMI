# coding: utf-8
# 2021/3/23 @ tongshiwei
import logging
from EduCDM import MIRT
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

train_data = pd.read_csv("E:\dataset\objectmath2015\Math1\\train_data.csv")
valid_data = pd.read_csv("E:\dataset\objectmath2015\Math1\\valid_data.csv")
test_data = pd.read_csv("E:\dataset\objectmath2015\Math1\\test_data.csv")

batch_size = 128


def transform(x, y, z, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.int64),
        torch.tensor(y, dtype=torch.int64),
        torch.tensor(z, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)


train, valid, test = [
    transform(data["user_id"], data["item_id"], data["score"], batch_size)
    for data in [train_data, valid_data, test_data]
]

logging.getLogger().setLevel(logging.INFO)

cdm = MIRT(4209, 15, 12)

cdm.train(train, valid, epoch=2)
cdm.save("mirt.params")

cdm.load("mirt.params")
auc, accuracy = cdm.eval(test)
print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))
