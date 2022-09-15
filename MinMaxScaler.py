# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def minmax_scaler(df,feature=None,f_range=(0,1)):
    """
    Created on Mon Sep  5 17:01:20 2022
        This function scales the sample data in the dataframe into a given range
    (default: 0,1). and return scaled data frame.

    Syntax     :  minmax_scaler(df,feature=None,f_range=(0,1)) 
    Parameters :
        df     : dataframe
        feature: desired feature to be scaled in the dataframe
                 (default = None) to scale all feature in dataframe
                 otherwise, scale only the given feature
        f_range: new range to be scaled into
    return     :
        scal_df: scaled data frame
        
    author: @Xero
    """
    
    scal_df = pd.DataFrame()
    old_range = df[feature].max()-df[feature].min()
    new_range = f_range[1]-f_range[0]
    new_val_arr = (((df[feature]-df[feature].min())*new_range/old_range))+f_range[0]
    scal_df = pd.DataFrame({feature: new_val_arr})
    return scal_df
