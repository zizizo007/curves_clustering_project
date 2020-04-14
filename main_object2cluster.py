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
        # X,X_norm - the data and the normlized data (according to z-score) are added as an array
        # X_df,X_norm_df - the data and the normlized data are added as a dataframe
        
        #create empty arrays for the loaded data
        self.X      = []
        self.X_norm = []
        
        #first, go over the source folder and load the data into a matrix X
        if typ==1:
            
            self.voltages_a = []    
            for voltage in np.arange(-2,2.01,0.01):
                self.voltages_a.append(str(round(voltage,2)))
            
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
            #convert the list to a numpy array
            self.X = np.transpose(np.array(self.X))
            self.X_norm = np.transpose(np.array(self.X_norm))
            #Convert the lists to a dataframe
            self.X_df       = pd.DataFrame(self.X, index=self.voltages_a)
            self.X_norm_df  = pd.DataFrame(self.X_norm, index=self.voltages_a)
            print('New data_2_cluster object created\n=================================\n')
            print('Object dimensions:\n')
            print('Number of sample types: ' + str(self.X.shape[0])+'\n')
            print('Number of samples taken: ' + str(self.X.shape[1])+'\n')
        
    def plot(self):
        #This function plots the object attributes according to the user selection
        selections = input('What would you like to plot?\nChoose from: X,X_norm\n')
        selections = selections.split(',')
        if 'X' in selections:
            plt.figure(1)
            print('Plotting the matrix X:\n')
            for column in self.X.T:
                plt.plot(self.voltages_a,self.X)
        if 'X_norm' in selections:
            plt.figure(2)
            print('Plotting the normalized matrix X:\n')
            for column in self.X_norm.T:
                plt.plot(self.voltages_a,self.X_norm)
