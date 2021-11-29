# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 09:15:33 2021

@author: AA
"""

from random import choice
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from numpy.linalg import pinv
import scipy.io as io
import csv
from sklearn import metrics
from sklearn.decomposition import PCA#pca降维
from sklearn.preprocessing import StandardScaler      #数据预处理
import random
class RELM_HiddenLayer:
    """
        正则化的极限学习机
        :param x: 初始化学习机时的训练集属性X
        :param num: 学习机隐层节点数
        :param C: 正则化系数的倒数
    """

    def __init__(self, x, num, C=10):
        row = x.shape[0]
        columns = x.shape[1]
        rnd = np.random.RandomState()
        # 权重w
        self.w = rnd.uniform(-1, 1, (columns, num))
        # 偏置b
        self.b = np.zeros([row, num], dtype=float)
        for i in range(num):
            rand_b = rnd.uniform(-0.4, 0.4)
            for j in range(row):
                self.b[j, i] = rand_b
        self.H0 = np.matrix(self.sigmoid(np.dot(x, self.w) + self.b))
        self.C = C
        self.P = (self.H0.H * self.H0 + len(x) / self.C).I
        #.T:共轭矩阵,.H:共轭转置,.I:逆矩阵

    @staticmethod
    def sigmoid(x):
        """
            激活函数sigmoid
            :param x: 训练集中的X
            :return: 激活值
        """
        return 1.0 / (1 + np.exp(-x))

   
    def Iteration(self,H,T):
        m,n=H.shape
        m,k=T.shape
        # X0=np.random.randint(-0.2,0.2,(n,k))############################
        y = [ random.uniform(-0.5,0.5) for i in range(n*k)] 

        X0 = np.array(y).reshape(n,k)
        
        alphamax = 1
        alphamin = 0.0001
        beta = 1
        gamma = 3
        delta = 1
        alpha = 1
        fc=2
        start = time.time()
        
        for j in range(1000):
            if j%100 == 0:
                print(j,time.time()-start)
            if(alpha<alphamin):
                alpha = alphamax
           
            Xet=self.ET(X0,gamma,1,H,T)
            if((Xet==X0).all()==False):
                X0=self.TT(X0,Xet,beta,1,H,T)
            else:
                X0=Xet
            Xrt=self.RT(X0,alpha,10,H,T)
            if((Xrt==X0).all()==False):
                X0=self.TT(X0,Xrt,beta,10,H,T)
            else:
                X0=Xrt
            Xat=self.AT(X0,delta,60,H,T)
            if((Xat==X0).all()==False):
                X0=self.TT(X0,Xat,beta,60,H,T)
            else:
                X0=Xat
            beta=beta/fc
            alpha = alpha/fc
        
        return X0
        #x0表示随机生成的初始矩阵
    def RT(self,x0,alpha,se,H,T):
        n, k = x0.shape
        demo = x0
        for i in range (se):
            R=np.random.uniform(-1, 1, [n, n])
            R1=np.dot(R,x0)
            P=np.linalg.norm(x0)
            M=n*P
            L=1/M
            T=alpha*L
            R2=T*R1
            newtest=x0+R2
            A=self.cal_l2(np.dot(H,newtest), T)
            B=np.linalg.norm(newtest)
            
            C=self.cal_l2(np.dot(H,demo), T)
            D=np.linalg.norm(demo)
            if(A+B<C+D):
                demo=x0+R2
    
        return demo
    
    def TT(self,x0,x1,beta,se,H,T):#TT里面x1是当前解
        n, k = x0.shape
        demo = x1
        for i in range(se):
            R=random.randint(0, 1) 
            R1=x1-x0
            
            L=np.linalg.norm(x1-x0)
           
            G=beta*R
            T=G/L
            
            R2=T*R1
            newtest=x1+R2
            
            A=self.cal_l2(np.dot(H,newtest), T)
            B=np.linalg.norm(newtest)
            
            C=self.cal_l2(np.dot(H,demo), T)
            D=np.linalg.norm(demo)
            
            if(A+B<C+D):
                demo=x1+R2
                
        return demo
    
    def ET(self,x0,gamma,se,H,T):
        n, k = x0.shape
        demo = x0
        for i in range(se):
            a=np.random.randn(n) 
            R=np.diag(a)
            R1=np.dot(R,x0)
            
            R2=gamma*R1
            newtest=x0+R2
            
            A=self.cal_l2(np.dot(H,newtest), T)
            B=np.linalg.norm(newtest)
            
            C=self.cal_l2(np.dot(H,demo), T)
            D=np.linalg.norm(demo)
            
            if(A+B<C+D):
                demo=x0+R2
           
        return demo


    def AT(self,x0,delta,se,H,T):
        n, k = x0.shape
        demo=x0
        for i in range(se):
            R=np.zeros((n,n))
            list=np.random.randn(n)
            i=np.random.randint(0,n)
            R[i][i]=choice(list)
            R1=np.dot(R,x0)
            
            R2=delta*R1
            newtest=x0+R2
            
            A=self.cal_l2(np.dot(H,newtest), T)
            B=np.linalg.norm(newtest)
            
            C=self.cal_l2(np.dot(H,demo), T)
            D=np.linalg.norm(demo)
            
            if(A+B<C+D):
                demo=x0+R2
          
        return demo
    def cal_l2(self,x1, x2):
        return np.linalg.norm(x1-x2)
    
    def softplus(x):
        """
            激活函数 softplus
            :param x: 训练集中的X
            :return: 激活值
        """
        return np.log(1 + np.exp(x))

    @staticmethod
    def tanh(x):
        """
            激活函数tanh
            :param x: 训练集中的X
            :return: 激活值
        """
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    # 回归问题 训练
    def regressor_train(self, T):
        """
            初始化了学习机后需要传入对应标签T
            :param T: 对应属性X的标签T
            :return: 隐层输出权值beta
        """
        all_m = np.dot(self.P, self.H0.H)
        self.beta = np.dot(all_m, T)
        # self.beta = self.Iteration(self.H0,T)
        return self.beta

    # 回归问题 测试
    def regressor_test(self, test_x):
        """
            传入待预测的属性X并进行预测获得预测值
            :param test_x:被预测标签的属性X
            :return: 被预测标签的预测值T
        """
        b_row = test_x.shape[0]
        h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        
    def classification_test(self, test_x):
        """
            传入待预测的属性X并进行预测获得预测值
            :param test_x:被预测标签的属性X
            :return: 被预测标签的预测值T
        """
        b_row = test_x.shape[0]
        h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        temp_Y=np.zeros(np.shape(result))
        max_index=result.argmax(axis=1)
        for i in range(np.shape(result)[0]):
            index=max_index[i]
            temp_Y[i,index]=1
        true_Y=temp_Y.nonzero()[1]
        return true_Y


retinamnist_data = np.load('D:/git-medmnist/pathmnist.npz')
P_train = retinamnist_data['train_images'].reshape(89996,-1)
P_test = retinamnist_data['test_images'].reshape(7180,-1)
T_train = retinamnist_data['train_labels'].flatten()
T_test = retinamnist_data['test_labels'].flatten()
T_train1=np.eye(9)[T_train.astype('int32')]
T_test1=np.eye(9)[T_test.astype('int32')]



stdsc = StandardScaler()  
P_train,P_test = stdsc.fit_transform(P_train),stdsc.fit_transform(P_test) #归一化



relm = RELM_HiddenLayer(P_train,100)
relm.regressor_train(T_train1)
result = relm.classification_test(P_test)
acc=metrics.accuracy_score(y_true=T_test,y_pred=result)


   
print(acc)

