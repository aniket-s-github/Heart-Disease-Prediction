# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 19:44:22 2022

@author: Aniket Dilip Shinde
"""
import numpy as np

class LogisticReg:
    def __init__(self, lr_rate=0.01, no_of_iters=2000):
        self.lr = lr_rate
        self.n_iters = no_of_iters
        self.weights = None
        self.bias = None
    
    def fit(self, x_train, y_train):
        n_samples,n_features = x_train.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            linear_model = np.dot(x_train,self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            dw = (1/n_samples)*np.dot(x_train.T,(y_pred-y_train))
            db = (1/n_samples)*np.sum(y_pred-y_train)
            self.weights -= self.lr*dw
            self.bias -= self.lr*db
            
    def predict(self, x_test):
        linear_model = np.dot(x_test,self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        y_pred_cls=[1 if i>0.52 else 0 for i in y_pred]
        return np.array(y_pred_cls)
    
    def sigmoid(self, theta):
        return 1/(1 + np.exp(-theta))
    
def accuracy(y_true, y_pred):
    accuracy=np.sum(y_true==y_pred)/len(y_true)
    return accuracy