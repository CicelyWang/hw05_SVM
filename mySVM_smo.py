# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:44:36 2016

@author: 11521053 王晓捷
"""

import numpy as np
import time
import matplotlib.pyplot as plt

def linear_kernel(x1,x2):
    return np.dot(x1,x2.T)

def rbf_kernel(x1,x2, gamma =3):
    diff = x1-x2
    return np.exp(-1*gamma*np.dot(diff,diff.T))
    
def polynomial_kernel(x1, x2, p):
    return (1 + np.dot(x1, x2.T)) ** p
    
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
    elif kernelType == "Gaussian":
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

class mySVM_SMO:
    def __init__(self,samples, labels,C,tol,iterNum=100, kernel = ('linear',0)):
        self.train_x = samples     #each row stands for a sample
        self.train_y = labels        
        self.kernel = kernel
        self.iterNum = iterNum     # termination condition for iteration 
        self.C = C                 #slack variable 
        if self.C is not None:
            self.C = float(self.C)
        self.tol = tol
        self.sampleNum = np.shape(self.train_x)[0]
        self.alphas = np.mat(np.zeros((self.sampleNum,1)))   #Lagrange factors
        self.b = 0
        self.errorCache = np.mat(np.zeros((self.sampleNum,2)))
        self.kernelMat = calKernelMatrix(self.train_x,self.kernel)
        self.w = np.zeros((self.train_x.shape[1],1))
       # print "kernelMat:",self.kernelMat


    # get the SVM model according to samples and labels
    # this fucntion descibes the main procedure of the training 
    def train(self):
        #calculate training time
        startTime = time.time()
    
        #start training
        allSample = True
        numChanged = 0    # the number of update
        iter = 0
        while (iter < self.iterNum) and ((numChanged > 0 ) or allSample):
            numChanged = 0
        
            if allSample: #update alphas over all samples
                for i in xrange(self.sampleNum):
                    j,error_i,error_j = self.chooseAlpha_j(i)
                    if j == -1:
                        continue
                    numChanged += self.optimizeAlphaPair(i,j,error_i,error_j)
                    iter += 1
            
            else: #update alphas over non-bound samples(0 < alpha < C )            
                updateList = np.nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
            
                for i in updateList:
                    j,error_i,error_j = self.chooseAlpha_j(i)
                    if j == -1:
                        continue
                    numChanged += self.optimizeAlphaPair(i,j,error_i,error_j)
                    iter += 1
        
            if allSample:
                allSample = False
            elif numChanged == 0:
                allSample = True
            
      #  supportVectorsIndex = np.nonzero(self.alphas.A > 0)[0]
#        for i in supportVectorsIndex:  
#            s = self.alphas[i] * self.train_y[i]
#            ss = np.multiply(s, self.train_x[i, :].T)
           
           #print ss
        for i in range(self.sampleNum):
            self.w += np.multiply(self.alphas[i] * self.train_y[i],self.train_x[i, :].T)
    
        print 'SMO training time is: ',(time.time()-startTime),"s"
        return 
    

    def calError(self,k):  
#        ss = np.multiply((self.alphas).T, self.train_y)
#        f_old_k = float(ss * self.kernelMat[:,k] + self.b)
        f_old_k = float(np.multiply(self.alphas,self.train_y).T * self.kernelMat[:,k] ) + self.b
        return f_old_k - float(self.train_y[k])

    def updateError(self,k):
        error = self.calError(k)
        self.errorCache[k]  = [1,error]
        return 

    def chooseAlpha_j(self,i):
        error_i = self.calError(i)
        r_i = error_i * self.train_y[i]
        j = -1
        error_j = 0
        #check if the alpha violates the KKT condition
        ### check and pick up the alpha who violates the KKT condition
        ## satisfy KKT condition
        # 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)
        # 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)
        # 3) yi*f(i) <= 1 and alpha == C (between the boundary)
        ## violate KKT condition
        # because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so
        # 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct) 
        # 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)
        # 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized
        if((r_i < -1*self.tol and self.alphas[i] < self.C) or ( r_i > self.tol and self.alphas[i] > 0)):
            self.errorCache[i] = [1,error_i]            
            candidateList = np.nonzero(self.errorCache[:,0].A)[0]
            maxstep = 0
        
            if len(candidateList) > 1:#find the alpha with max step size
                for k in candidateList:
                    if i == k:
                        continue
                    error_k = self.calError(k)
                    if abs(error_k - error_i) > maxstep:
                        maxstep = abs(error_k - error_i)
                        error_j = error_k
                        j = k
            else :#select alpha j randomly through the entire training set
                k = i
                while k == i:
                    k = int(np.random.uniform(0,self.sampleNum))    
                j = k
                error_j = self.calError(j)
        return j,error_i,error_j


    #optimize alpha_i and alpha_j
    #modify b
    def optimizeAlphaPair(self,i,j,error_i,error_j):

        if i == j :
            return 0
    
        alpha_i_old = self.alphas[i].copy()
        alpha_j_old = self.alphas[j].copy()

        #calculate the boundary L and H for alpha_j
        s = self.train_y[i] * self.train_y[j]
        if s < 0 :# y_i != y_j
            L = float(np.max([0, self.alphas[j] - self.alphas[i]]))
            H = float(np.min([self.C, self.C + self.alphas[j] - self.alphas[i]]))
        else:
            L = float(np.max([0, self.alphas[j] + self.alphas[i] - self.C]))
            H = float(np.min([self.C, self.alphas[j] + self.alphas[i]]))
        if L == H :
            return 0

        eta = 2.0 * self.kernelMat[i,j] - self.kernelMat[i,i] - self.kernelMat[j,j]
        if eta < 0:
            temp_j = alpha_j_old - self.train_y[j] * (error_i - error_j)/eta
            #clip alpha_j        
            if temp_j > H:
                self.alphas[j] = H
            elif temp_j < L:
                self.alphas[j] = L
            else:
                self.alphas[j] = temp_j
        else:##################################qita
            return 0
        self.updateError(j)
        #if alpha_j not moving enough, just return        
        if abs(self.alphas[j] - alpha_j_old) < 0.0001: 
            return 0
    
        #update alpha_i 
        self.alphas[i] = alpha_i_old + s * (alpha_j_old - self.alphas[j])
        self.updateError(i)
    
        #update b
        b_i = self.b - error_i - self.train_y[i] * (self.alphas[i] - alpha_i_old) * self.kernelMat[i,i] \
                                    - self.train_y[j] * (self.alphas[j] - alpha_j_old) * self.kernelMat[i,j] 
        b_j = self.b - error_j - self.train_y[i] * (self.alphas[i] - alpha_i_old) * self.kernelMat[i,j] \
                                    - self.train_y[j] * (self.alphas[j] - alpha_j_old) * self.kernelMat[j,j]                  
        if (self.alphas[i] > 0) and (self.alphas[i] < self.C):
            self.b = b_i
        elif (self.alphas[j] > 0) and (self.alphas[j] < self.C):
            self.b = b_j
        else:
            self.b = (b_i + b_j)/2.0

        return 1


    def show(self):
        for i in xrange(self.sampleNum):
            if self.train_y[i] == -1:
                plt.plot(self.train_x[i,0],self.train_x[i,1],'or')
            elif self.train_y[i] == 1:
                plt.plot(self.train_x[i,0],self.train_x[i,1],'ob')
        
        #draw the classify line
        min_x = (min(self.train_x[:, 0]))
        max_x =(max(self.train_x[:, 0]))  
        y_min_x = float(-self.b - self.w[0] * min_x) / self.w[1]  
        y_max_x = float(-self.b - self.w[0] * max_x) / self.w[1]  
        plt.plot([float(min_x), float(max_x)], [y_min_x, y_max_x], '-g')  
        plt.show()  
      
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