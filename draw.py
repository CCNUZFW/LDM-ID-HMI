# import seaborn as sns
import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
# import pandas as pd
import re
file_name = './train.txt'
ls = []
loss = []
auc = []
acc = []
vauc = []
vacc = []
with open(file_name) as f:
    ls = f.readlines()
cnt = 0
for l in ls:
    matloss = re.search('loss : ', l)
    matvalid = re.search('valid auc : ', l)
    if matloss:
        start = matloss.span()[1]
        #print(l[start:start+7], l[start+15:start+22], l[start+35:start+42])
        loss.append(float(l[start:start+7]))
        auc.append(float(l[start+15:start+22]))
        acc.append(float(l[start+35:start+42]))
        cnt += 1
    if matvalid:
        start = matvalid.span()[1]
        vauc.append(float(l[start:start+7]))
        vacc.append(float(l[start+26:start+33]))
    # if mat:
    # print(l[mat[1]:])
a, b = np.array(vauc[150:]).max(), np.array(vauc[150:]).min()
print(a, b, (a+b)/2, (a-b)/2)

plt.figure(dpi=300)
#plt.plot(range(200), loss, label="training loss")
plt.plot(range(200), auc, label="training AUC")
plt.plot(range(200), acc, label="training accuracy")
plt.plot(range(200), vauc, label="validate AUC")
plt.plot(range(200), vacc, label="validate accuracy")
#plt.title("Training Process")
plt.xlabel("Epoch")
plt.ylabel("")
plt.legend()
plt.show()
