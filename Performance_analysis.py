# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 20:33:59 2022

@author: Xero
"""

import numpy as np

def mse(y1,y2):
    m = len(y1)
    err = np.multiply((y1-y2),(y1-y2))
    mean_sq_err = (1/m)*np.sum(err)
    return mean_sq_err

# R squared value evaluation function
def r_2(prediction,y):
    """ A function that compute R squared value of Regression Model.
    
        Parameters
        ----------
        prediction : 20 x 1
        y : 20 x 1
        
        Returns
        -------
        r2_value : 1 x 1"""
    
    # Initialize useful information
    # No. of outputs
    total = y.shape[0] 
    # Average of actual outputs
    y_avg = np.sum(y)/total
    # Sum of squared error(SSE)
    SSE = np.sum(np.square(y-prediction))
    # Total sum of squares(SSTO)
    SSTO = np.sum(np.square(y-y_avg))
    # R square value 
    r2_value = 1 - (SSE/SSTO)
    
    return r2_value 