import logging
import numpy as np
import json
from EduCDM import EMDINA as DINA
from EduData import get_data
#
#get_data("math2015", "../../../data")

q_m = np.loadtxt("E:\dataset\objectmath2015\Math1\\q_m.csv", dtype=int, delimiter=',')
prob_num, know_num = q_m.shape[0], q_m.shape[1]

# training data
with open("E:\dataset\objectmath2015\Math1\\train_data.json", encoding='utf-8') as file:
    train_set = json.load(file)
stu_num = max([x['user_id'] for x in train_set]) + 1
R = -1 * np.ones(shape=(stu_num, prob_num))
for log in train_set:
    R[log['user_id'], log['item_id']] = log['score']

# testing data
with open("E:\dataset\objectmath2015\Math1\\test_data.json", encoding='utf-8') as file:
    test_set = json.load(file)

logging.getLogger().setLevel(logging.INFO)

cdm = DINA(R, q_m, stu_num, prob_num, know_num, skip_value=-1)

cdm.train(epoch=2, epsilon=1e-3)
cdm.save("dina.params")

cdm.load("dina.params")
rmse, mae, auc = cdm.eval(test_set)
print("RMSE: %.6f, MAE: %.6f, AUC: %.6f" % (rmse, mae, auc))

# ---incremental training
# new_data = [{'user_id': 0, 'item_id': 0, 'score': 1.0}, {'user_id': 1, 'item_id': 2, 'score': 0.0}]
# cdm.inc_train(new_data, epoch=2, epsilon=1e-3)

# ---evaluate user's state
# stu_rec = np.array([0, 1, -1, 0, -1, 0, 1, 1, 0, 1, 0, 1, 0, -1, -1, -1, -1, 0, 1, -1])
# dia_id, dia_state = cdm.transform(stu_rec)
# print("id of user's state is %d, state is " % dia_id + str(dia_state))


# import numpy as np
# import random
# import json
#
# train_ratio = 0.8
# valid_ratio = 0
# # Q matrix
# np.savetxt("E:\dataset\objectmath2015\Math1\\q_m.csv", np.loadtxt("E:\dataset\objectmath2015\Math1\\q.txt", dtype=int), delimiter=',', fmt='%d')
#
# # response matrix, split dataset
# R = (np.loadtxt("E:\dataset\objectmath2015\Math1\\data.txt") == 1).astype(float)
# R = R[:, 0:15]
#
# stu_num, prob_num = R.shape[0], R.shape[1]
# train_logs, valid_logs, test_logs = [], [], []
# for stu in range(stu_num):
#     stu_logs = []
#     for prob in range(prob_num):
#         log = {'user_id': int(stu), 'item_id': int(prob), 'score': R[stu][prob]}
#         stu_logs.append(log)
#     random.shuffle(stu_logs)
#     train_logs += stu_logs[: int(train_ratio * prob_num)]
#     valid_logs += stu_logs[int(train_ratio * prob_num): int(train_ratio * prob_num) + int(valid_ratio * prob_num)]
#     test_logs += stu_logs[int(train_ratio * prob_num) + int(valid_ratio * prob_num):]
#
# with open("E:\dataset\objectmath2015\Math1\\train_data.json", 'w', encoding='utf8') as file:
#     json.dump(train_logs, file, indent=4, ensure_ascii=False)
# with open("E:\dataset\objectmath2015\Math1\\valid_data.json", 'w', encoding='utf8') as file:
#     json.dump(valid_logs, file, indent=4, ensure_ascii=False)
# with open("E:\dataset\objectmath2015\Math1\\test_data.json", 'w', encoding='utf8') as file:
#     json.dump(test_logs, file, indent=4, ensure_ascii=False)
#
# print(train_logs[0], test_logs[0])