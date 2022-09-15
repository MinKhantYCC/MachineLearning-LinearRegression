# -*- coding: utf-8 -*-

import numpy as np
from hypothesis import hyp

def cost_fun(X,y,theta,lamb=0):
    """
     This function calculates the cost function (the mean square error function) 
    with regularization in machine learning. The function
    requires X (mxn), theta(n+1xk), y(mxk). Also
    the hypothesis function is required before we use this function.

    Syntax    : cost(X,y,theta,alpha=1,lamb=0)
    Parameters:
        X: array-like parameters used for input features. (m*n) dimension
        y: array-like parameters of ouput target class. (m*k) dimension
        theta: array-like parameters of weight ([n+1]*k) dimension

    return:
        J: cost error of the algorithm
     
    Created on Tue Sep  6 12:58:28 2022

    @author: Xero
    """
    
    #useful variables
    #size
    m,n = X.shape
    
    y_pred = hyp(X,theta)                          #predict target value
    err = y_pred - y                               #calculate error
    regu = (lamb/(2*m))*(np.sum(np.square(theta[1:n]))) #calculate regularization
    J = (1/(2*m))*np.sum(np.square(err))+regu      #calculate cost error
    return J