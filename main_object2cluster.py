# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import linregress
import scipy.signal as sig
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from scipy.stats import zscore
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class data_2_cluster(object):
    
    def __init__(self, path, typ=1):
        # when initializing the main object:
        # a - the data and the normlized data (according to z-score) are added as an array
        # b - the data and the normlized data are added as a dataframe
        
        #create empty arrays for the loaded data
        self.X      = []
        self.X_norm = []
        
        voltages_a = []    
        for voltage in np.arange(-2,2.01,0.01):
            voltages_a.append(str(round(voltage,2)))
    
        
        #first, go over the source folder and load the data into a matrix X
        if type==1:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if os.path.splitext(file)[1] == '.xlsx':
                        print('Working On: \n')
                        print(os.path.join(root,file) + '\n')
                        df = pd.read_excel(os.path.join(root,file))
                        for column in df:
                            if(df[column].name == 'Voltage'):
                                continue
                            else:
                                # I(V) raw data
                                self.X.append(np.asarray(df[df[column].name])) 
                                # I(V) normalized
                                self.X_norm.append(np.asarray(zscore(df[df[column].name])))
            #Convert the lists to a dataframe
            self.X_df       = pd.DataFrame(np.asarray(self.X), columns=voltages_a)
            self.X_norm_df  = pd.DataFrame(np.asarray(self.X_norm), columns=voltages_a)
