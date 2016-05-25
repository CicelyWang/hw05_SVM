# -*- coding: utf-8 -*-
"""
Created on Wed May 18 10:26:13 2016

@author: 11521053 王晓捷
"""

import mySVM_cvxopt as cSVM
import matplotlib.pyplot as plt
import mySVM_smo as sSVM
import numpy as np
import math
import random

def gen_non_lin_separable_data():
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]
    cov = [[1.0,0.8], [0.8, 1.0]]
    X1 = np.random.multivariate_normal(mean1, cov, 50)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 50)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
    y2 = np.ones(len(X2)) * -1

    #同心半圆 ，X1半径为1， X2半径为4，圆心均为圆点
#    X11 = random.sample(range(-100,100),200)
#    X11 = np.multiply(X11 ,0.01)
#    X12 = []
#    X1 = []
#    for i in range(200):
#        X12.append(math.sqrt(1-X11[i]*X11[i]))
#    X12_noise = [np.random.normal(0,0.05) + x12 for x12 in X12]
#    for i in range(200):
#        X1.append([X11[i],X12_noise[i]])
#    y1 = np.ones(len(X1))
#    
#    X21 = random.sample(range(-400,400),200)
#    X21 = np.multiply(X21 , 0.01)
#    X2 = []
#    X22 = []
#    for i in range(200):
#        X22.append(math.sqrt(16-X21[i]*X21[i]))
#    X22_noise = [np.random.normal(0,0.05) + x22 for x22 in X22]
#    for i in range(200):
#        X2.append([X21[i],X22_noise[i]])
#    y2 = np.ones(len(X1)) * -1

    return X1, y1, X2, y2
    
def gen_lin_separable_data():
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[0.8, 0.6], [0.6, 0.8]])
    X1 = np.random.multivariate_normal(mean1, cov, 100)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 100)
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2

def split_train(X1, y1, X2, y2):
    num = len(y1)
    train_num = int(num * 0.8)
    X1_train = X1[:train_num]
    y1_train = y1[:train_num]
    X2_train = X2[:train_num]
    y2_train = y2[:train_num]
    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))
    return X_train, y_train

def split_test(X1, y1, X2, y2):
    num = len(y1)
    test_num = int(num * 0.2)
    X1_test = X1[test_num:]
    y1_test = y1[test_num:]
    X2_test = X2[test_num:]
    y2_test = y2[test_num:]
    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))
    return X_test, y_test        
        
def show(train_x,train_y):
    for i in xrange(len(train_y)):
        if train_y[i] == -1:
            plt.plot(train_x[i,0],train_x[i,1],'or')
        elif train_y[i] == 1:
            plt.plot(train_x[i,0],train_x[i,1],'ob')
    
    plt.show()
            
if __name__ == "__main__":
    
    X1,y1,X2,y2 = gen_lin_separable_data()
    train_x,train_y = split_train(X1,y1,X2,y2)
    test_x,test_y = split_test(X1,y1,X2,y2)
    C = 0.5
    tol = 0.001
    maxIter = 50
    csvm_linear = cSVM.MySVM(train_x,train_y,C,kernel = ("linear",0))
    csvm_linear.train()
    csvm_linear.show()
    csvm_linear.test(test_x,test_y)
    ssvm_linear = sSVM.mySVM_SMO(np.mat(train_x),np.mat(train_y).transpose(),C,tol,maxIter,kernel = ("linear",0))
    ssvm_linear.train()
    ssvm_linear.show()
    ssvm_linear.test(test_x,test_y)
    
    #非线性可分
    X1, y1, X2, y2   =gen_non_lin_separable_data()
    train_x,train_y = split_train(X1,y1,X2,y2)
    test_x,test_y = split_test(X1,y1,X2,y2)    
    C = 0.5
    tol = 0.001
    maxIter = 50
    csvm_klinear = cSVM.MySVM(train_x,train_y,C,kernel = ("linear",3.0))
    show(train_x,train_y)
    csvm_klinear.train()
    csvm_klinear.test(test_x,test_y)
    ssvm_klinear = sSVM.mySVM_SMO(np.mat(train_x),np.mat(train_y).transpose(),C,tol,maxIter,kernel = ("linear",3))
    ssvm_klinear.train()
    ssvm_klinear.test(test_x,test_y)
    
    csvm_krbf = cSVM.MySVM(train_x,train_y,C,kernel = ("rbf",1.0))
    show(train_x,train_y)
    csvm_krbf.train()
    csvm_krbf.test(test_x,test_y)
    ssvm_krbf = sSVM.mySVM_SMO(np.mat(train_x),np.mat(train_y).transpose(),C,tol,maxIter,kernel = ("rbf",1.0))
    ssvm_krbf.train()
    ssvm_krbf.test(test_x,test_y)
    
    csvm_kpoly = cSVM.MySVM(train_x,train_y,C,kernel = ("polynomial",4))
    show(train_x,train_y)
    csvm_kpoly.train()
    csvm_kpoly.test(test_x,test_y)
    ssvm_kpoly = sSVM.mySVM_SMO(np.mat(train_x),np.mat(train_y).transpose(),C,tol,maxIter,kernel = ("polynomial",4))
    ssvm_kpoly.train()
    ssvm_kpoly.test(test_x,test_y)
    
    
    csvm_kgs = cSVM.MySVM(train_x,train_y,C,kernel = ("Gaussian",2.0))
    show(train_x,train_y)
    csvm_kgs.train()
    csvm_kgs.test(test_x,test_y)
    ssvm_kgs = sSVM.mySVM_SMO(np.mat(train_x),np.mat(train_y).transpose(),C,tol,maxIter,kernel = ("Gaussian",2.0))
    ssvm_kgs.train()
    ssvm_kgs.test(test_x,test_y)
