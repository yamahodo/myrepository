from os import error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import codecs
from scipy.sparse import csr_matrix

#读取数据
data_users = []
data_train = []
data_test = []
data_movie = []
with open('D:\\Work\\code\\test\\hw2\\Project2-data\\users.txt', 'r') as f_input:
    for line in f_input:
        data_users.append(line.strip().split(' ',0))
with open('D:\Work\code\\test\hw2\Project2-data\\netflix_train.txt', 'r') as f_input:
    for line in f_input:
        data_train.append(list(line.strip().split(' ')))
with open('D:\Work\code\\test\hw2\Project2-data\\netflix_test.txt', 'r') as f_input:
    for line in f_input:
        data_test.append(list(line.strip().split(' ')))
with open('D:\\Work\\code\\test\\hw2\\Project2-data\\movie_titles.txt', 'r',encoding='ISO-8859-1',) as f_input:
    for line in f_input:
        data_movie.append(list(line.strip().split(',',2)))
data_movie = np.array(data_movie)
data_users = np.array(data_users)
data_train = np.array(data_train)
data_test = np.array(data_test)

# print(data_users.shape)
# print(data_movie.shape)

#数据预处理
X_train = np.zeros((data_users.shape[0],data_movie.shape[0]))
A = np.zeros((data_users.shape[0],data_movie.shape[0]))
X_test = np.zeros((data_users.shape[0],data_movie.shape[0]))

dictusers = {}
for i in range(data_users.shape[0]):
    a = str(int(data_users[i]))
    dict1 = {a:i}
    dictusers.update(dict1)
# print(dictusers)
# userid = '251613'
# print(dictusers[userid])
for i in range(data_train.shape[0]):
    userid = data_train[i][0]
    movieid = data_train[i][1]
    score = data_train[i][2]
    usernum = dictusers[userid]
    X_train[int(usernum)][int(movieid)-1] = score
    if int(score) > 0 :
        A[int(usernum)][int(movieid)-1] = 1
for i in range(data_test.shape[0]):
    userid = data_test[i][0]
    movieid = data_test[i][1]
    score = data_test[i][2]
    usernum = dictusers[userid]
    # print(usernum,'--',movieid,'--',score)
    X_test[int(usernum)][int(movieid)-1] = score
X_train_sparse = csr_matrix(X_train)
X_test_sparse = csr_matrix(X_test)
#协同过滤

#定义sim函数
def sim(x,y):
    xy = sum(x*y)
    absx = sum(x*x)**0.5
    absy = sum(y*y)**0.5
    return xy/(absx*absy)
#判断用户i是否喜欢电影j
def predictscore(i,j,X_train_sparse):
    fenzi = 0
    fenmu = 0
    for k in range(data_users.shape[0]):
        if X_train_sparse[k,j] > 0:
            fenzi = fenzi +sim(X_train[i,],X_train[k,])*X_train_sparse[k,j]
            fenmu = fenmu + sim(X_train[i,],X_train[k,])
    return fenzi/fenmu

def calculatermse(X_train,X_test,X_train_sparse):
    X_pre = X_train
    m,n=X_train.shape
    for i in range(X_pre.shape[0]):
        for j in range(X_pre.shape[1]):
            print(i,j)
            if X_train_sparse[i,j] == 0:
                X_pre[i,j] = predictscore(i,j,X_train_sparse)
    err = X_test - X_pre
    RMSE = np.linalg.norm(err)/np.sqrt(m*n)
    print('RMSE = ',RMSE)
#基于梯度下降的矩阵分解算法
def gradAscent(X_train,X_test,alpha,k,p,A,epochs):
    #1、初始化UV
    m,n=X_train.shape
    U = np.mat(np.random.random((m, k)))
    V = np.mat(np.random.random((k, n)))
    X_test = np.mat(X_test)
    RMSE_metric = []
    J_metric = []
    for epoch in epochs:
        #2计算梯度
        A = np.mat(A)
        X = np.mat(X_train)
        a = U*V.I - X 
        piandaoU = np.multiply(A,a)*V + 2*p*U
        piandaoV = np.multiply(A,a)*U + 2*p*V
        #3、更新UV
        U = U - alpha*piandaoU
        V = V - alpha*piandaoV
        #4目标函数值 RMSE
        J = 0.5*np.linalg.norm(np.multiply(A,(-a)))+p*np.linalg.norm(U)+p*np.linalg.norm(V)
        X_pre = U*V
        err = X_test - X_pre
        RMSE = np.linalg.norm(err)/np.sqrt(m*n)
        RMSE_metric.append(RMSE)
        J_metric.append(J)
        print("训练次数=",epoch,'J = ',J,'RMSE = ',RMSE)
        print('-------------------')
    plt.plot(epochs, J_metric, 'bo--')
    plt.plot(epochs, RMSE_metric, 'ro--')
    plt.xlabel("Epochs")
    plt.ylabel('metrics')
    plt.legend(["J", 'RMSE'])

calculatermse(X_train,X_test,X_train_sparse)