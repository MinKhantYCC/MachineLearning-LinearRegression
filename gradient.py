# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 17:12:05 2022

@author: Xero
"""
import numpy as np
from CostFun import cost_fun
from scipy.optimize import fmin_cg
from hypothesis import hyp

def grad(X,y,theta, alpha=1, lamb=0, max_iter = 100):
    #size
    m,n = X.shape   #no of samples & no of features
    k = y.shape[1]  #no of class
    
    args = (X,y,lamb)
    
    J_hist = list()
    for i in range(1,max_iter+1):
        J_hist.append(cost_fun(X, y, theta))
        theta = theta - (alpha/m)*np.dot(X.T,(hyp(X,theta)-y))
        theta[1:n] = theta[1:n] + (alpha*lamb/m)*theta[1:n]
    return J_hist, theta