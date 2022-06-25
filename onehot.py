import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

theta = pd.read_csv('E:\\neuralCDM1\\data\\theta.csv', header=None)
theta = theta.values
skills = pd.read_csv('E:\\neuralCDM1\\data\\est_skills.csv', header=None)
skills = skills.values
feature = pd.read_csv('E:\\neuralCDM3\\data\\feature.csv', header=None)
feature = feature.values

d = 0.85
for i in range(15):
      if feature[i][3] > d:
        d = feature[i][3]
print(d)

# for i in range(4209):
#     if (theta[0][i] > -0.5) and (theta[0][i]<-0.4):
#         theta[0][i] = 0
#     elif (theta[0][i]>-0.4) and (theta[0][i] < -0.3):
#         theta[0][i] = 1
#     elif (theta[0][i]>-0.3) and (theta[0][i]<-0.2):
#         theta[0][i] = 2
#     elif (theta[0][i]>-0.2) and (theta[0][i]<-0.1):
#         theta[0][i] = 3
#     elif (theta[0][i]>-0.1) and (theta[0][i]<0):
#         theta[0][i] = 4
#     elif (theta[0][i]>0) and (theta[0][i]<0.1):
#         theta[0][i] = 5
#     elif (theta[0][i]>0.1) and (theta[0][i]<0.2):
#         theta[0][i] = 6
#     elif (theta[0][i]>0.2) and (theta[0][i]<0.3):
#         theta[0][i] = 7
#     elif (theta[0][i] > 0.3) and (theta[0][i]<0.4):
#         theta[0][i] = 8
#
# theta.to_csv('E:\\neuralCDM3\\data\\theta_lisan.csv',header=None)
# theta = OneHotEncoder(sparse = False).fit_transform(theta)
#
# np.save('E:\\neuralCDM3\\data\\theta_onehot.npy', theta)
# print(theta)

# for i in range(4209):
#     for j in range(11):
#         if skills[i][j]>=0 and skills[i][j]<0.1:
#             skills[i][j]=0
#         if skills[i][j]>=0.1 and skills[i][j]<0.2:
#             skills[i][j]=1
#         if skills[i][j]>=0.2 and skills[i][j]<0.3:
#             skills[i][j]=2
#         if skills[i][j]>=0.3 and skills[i][j]<0.4:
#             skills[i][j]=3
#         if skills[i][j]>=0.4 and skills[i][j]<0.5:
#             skills[i][j]=4
#         if skills[i][j]>=0.5 and skills[i][j]<0.6:
#             skills[i][j]=5
#         if skills[i][j]>=0.6 and skills[i][j]<0.7:
#             skills[i][j]=6
#         if skills[i][j]>=0.7 and skills[i][j]<0.8:
#             skills[i][j]=7
#         if skills[i][j]>=0.8 and skills[i][j]<0.9:
#             skills[i][j]=8
#         if skills[i][j]>=0.9 and skills[i][j]<=1:
#             skills[i][j]=9
# print(skills)
# skills = OneHotEncoder(sparse = False).fit_transform(skills)
# np.save('E:\\neuralCDM3\\data\\skills_onehot.npy', skills)
# print(skills.shape)

# feature = OneHotEncoder(sparse = False).fit_transform(feature)
# np.save('E:\\neuralCDM3\\data\\feature_onehot.npy', feature)

skills = np.load('E:\\neuralCDM3\\data\\onehot\\skills_onehot.npy')    #4209,99
feature = np.load('E:\\neuralCDM3\\data\\onehot\\feature_onehot.npy')  #15,60
theta = np.load('E:\\neuralCDM3\\data\\onehot\\theta_onehot.npy')      #4209,5
print(skills.shape,theta.shape,feature.shape)
