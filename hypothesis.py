# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def hyp(X,theta):
    """
        This function is used to calculate hypothesis function in Linear Regression.

    Syntax    : hyp(X,theta)
    Parameters:
        X     : array like input data
        theta : (number_of_samples*1) array like weight
    return    :
        y     : (number_of_samples*1)

    Created on Mon Sep  5 18:52:39 2022

    @author: Xero
    """
    
    #size
    m,n = X.shape
    _,k = theta.shape
    y = np.zeros((m,k))
    y = np.dot(X,theta)
    return y
