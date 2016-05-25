# -*- coding: utf-8 -*-
"""
Created on Thu May 12 09:56:45 2016

@author: Cicely
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxopt
import time

zero = 1e-3

def linear_kernel(x1,x2):
    return np.dot(x1,x2)

def rbf_kernel(x1,x2, gamma =3):
    diff = x1-x2
    return np.exp(-1*gamma*np.dot(diff,diff.T))

def polynomial_kernel(x1, x2, p):
    return (1 + np.dot(x1, x2)) ** p
    
def gaussian_kernel(x1, x2, sigma):
    return np.exp(-np.linalg.norm(x1-x2)**2 / (2 * (sigma ** 2)))
    
def calKernelValue(x1,x2,kernel):
    kernelType = kernel[0]
    kernelValue = 0
    if kernelType == "linear":
        kernelValue = linear_kernel(x1,x2)
    elif kernelType == "rbf":
        gamma = kernel[1]
        if gamma <= 0:
            gamma = 1.0
        kernelValue = rbf_kernel(x1,x2,gamma)
    elif kernelType == "polynomial":
        p = kernel[1]
        if p <= 0:
            p = 3
        kernelValue = polynomial_kernel(x1,x2,p)
    elif kernelType == "Guassian":
        sigma = kernel[1]
        if sigma <= 0:
            sigma = 5.0
        kernelValue = gaussian_kernel(x1,x2,sigma)
    return kernelValue
    
def calKernelMatrix(train_x,kernel):
    num = len(train_x)
    kernelMat = np.mat(np.zeros((num,num)))
    for i in xrange(num):
        for j in xrange(num):
            kernelMat[i,j] = calKernelValue(train_x[i],train_x[j],kernel)
    return kernelMat
    

class MySVM:
    def __init__(self,sample,label,C,kernel = ("linear",0)):
        self.train_x = sample
        self.train_y = label
        self.kernel = kernel
        self.nfeature = self.train_x.shape[1]
        self.sampleNum = len(label)
        self.C = C
        if self.C is not None:
            self.C = float(self.C)
        self.kernelMat = calKernelMatrix(self.train_x,kernel)
    
    def train(self):
        startTime = time.time()
        P = cvxopt.matrix(np.multiply(np.dot(self.train_y,self.train_y),self.kernelMat))
        q = cvxopt.matrix(np.ones((self.sampleNum,1))*-1)
        A = cvxopt.matrix(self.train_y,(1,self.sampleNum))
        b = cvxopt.matrix(0.0)
        
        if self.C is None:# no outliers
            G = cvxopt.matrix(np.diag(np.ones(self.sampleNum) * -1))
            h = cvxopt.matrix(np.zeros(self.sampleNum,1))
        else:
            G1 = np.diag(np.ones(self.sampleNum) * -1)
            G2 = np.identity(self.sampleNum)
            G = cvxopt.matrix(np.vstack((G1,G2)))
            h1 = np.zeros((self.sampleNum,1))
            h2 = np.ones((self.sampleNum,1)) * self.C
            h = cvxopt.matrix(np.vstack((h1,h2)))
        
        #solve QP problem
        result = cvxopt.solvers.qp(P,q,G,h,A,b)
        self.alphas = np.mat(result['x'])
        
        #calculate w according to alphas
        self.w =  np.zeros((self.nfeature,1))
        supportVectorsIndex = np.nonzero(self.alphas.A > 0)[0]        
        for i in supportVectorsIndex:
            s = self.alphas[i] * self.train_y[i]
            ss = np.multiply(s, self.train_x[i, :])
            self.w += np.array(ss.T)
        
        #calculate b according w
        wx_neg = []
        wx_pos = []
        for i in xrange(self.sampleNum):
            if self.train_y[i] == 1:
                wx_pos.append( np.dot(self.w.T, self.train_x[i].T))
            elif self.train_y[i] == -1:
                wx_neg.append( np.dot(self.w.T, self.train_x[i].T))
                      
        self.b = -1 * (max(wx_neg) + min(wx_pos)) / 2.0
        
        print "CVXOPT training time is :" , time.time()-startTime,"s"
        
        
    def test(self,test_x,test_y):
        num = len(test_y)
        supportVectorsIndex = np.nonzero(self.alphas.A > 0)[0]
        matchCount = 0  
        for i in xrange(num):  
            predict = 0
            for k in supportVectorsIndex:
                kernelValue = calKernelValue(self.train_x[k],test_x[i],self.kernel)
                predict += self.alphas[k]*self.train_y[k]*kernelValue
            predict += self.b
            if np.sign(predict) == np.sign(test_y[i]):  
                matchCount += 1  
        accuracy = float(matchCount) / num 
                
        print  "accuracy: ",accuracy
        print "kernel:",self.kernel
        
    def show(self):
        for i in xrange(self.sampleNum):
            if self.train_y[i] == -1:
                plt.plot(self.train_x[i,0],self.train_x[i,1],'or')
            elif self.train_y[i] == 1:
                plt.plot(self.train_x[i,0],self.train_x[i,1],'ob')
        
        
        #draw the classify line
        min_x = min(self.train_x[:, 0])
        max_x = max(self.train_x[:, 0])  
        y_min_x = float(-self.b - self.w[0] * min_x) / self.w[1]  
        y_max_x = float(-self.b - self.w[0] * max_x) / self.w[1]  
        plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')  
        plt.show()  
       
        
        


