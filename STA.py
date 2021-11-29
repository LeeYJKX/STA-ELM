# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 09:15:33 2021

@author: AA
"""

from random import choice
import numpy as np
import pandas as pd
import time
import random


   
def Iteration(H,T):
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
    
    for j in range(5000):
        if j%100 == 0:
            print(j,time.time()-start)
        if(alpha<alphamin):
            alpha = alphamax
       
        Xet=ET(X0,gamma,1,H,T)
        if((Xet==X0).all()==False):
            X0=TT(X0,Xet,beta,1,H,T)
        else:
            X0=Xet
        Xrt=RT(X0,alpha,10,H,T)
        if((Xrt==X0).all()==False):
            X0=TT(X0,Xrt,beta,10,H,T)
        else:
            X0=Xrt
        Xat=AT(X0,delta,60,H,T)
        if((Xat==X0).all()==False):
            X0=TT(X0,Xat,beta,60,H,T)
        else:
            X0=Xat
        beta=beta/fc
        alpha = alpha/fc
    
    return X0
    #x0表示随机生成的初始矩阵
def RT(x0,alpha,se,H,T):
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
        A=cal_l2(np.dot(H,newtest), T)
        B=np.linalg.norm(newtest)
        
        C=cal_l2(np.dot(H,demo), T)
        D=np.linalg.norm(demo)
        if(A+B<C+D):
            demo=x0+R2

    return demo

def TT(x0,x1,beta,se,H,T):#TT里面x1是当前解
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
        
        A=cal_l2(np.dot(H,newtest), T)
        B=np.linalg.norm(newtest)
        
        C=cal_l2(np.dot(H,demo), T)
        D=np.linalg.norm(demo)
        
        if(A+B<C+D):
            demo=x1+R2
            
    return demo

def ET(x0,gamma,se,H,T):
    n, k = x0.shape
    demo = x0
    for i in range(se):
        a=np.random.randn(n) 
        R=np.diag(a)
        R1=np.dot(R,x0)
        
        R2=gamma*R1
        newtest=x0+R2
        
        A=cal_l2(np.dot(H,newtest), T)
        B=np.linalg.norm(newtest)
        
        C=cal_l2(np.dot(H,demo), T)
        D=np.linalg.norm(demo)
        
        if(A+B<C+D):
            demo=x0+R2
       
    return demo


def AT(x0,delta,se,H,T):
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
        
        A=cal_l2(np.dot(H,newtest), T)
        B=np.linalg.norm(newtest)
        
        C=cal_l2(np.dot(H,demo), T)
        D=np.linalg.norm(demo)
        
        if(A+B<C+D):
            demo=x0+R2
      
    return demo
def cal_l2(x1, x2):
    return np.linalg.norm(x1-x2)

