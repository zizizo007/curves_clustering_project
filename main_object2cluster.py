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
        selections = input('What would you like to plot?\nChoose from: X,X_norm,X_shifted\n')
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
        if 'X_shifted' in selections:
            plt.figure(3)
            print('Plotting the normalized shifted matrix X:\n')
            for column in self.X_shifted.T:
                plt.plot(self.voltages_a,self.X_shifted)
    
    def pca(self, norm=True, shiftData=True):
        #this function calculaation the principal components of the data
        
        if norm:
            data = self.X_norm
        else:
            data = self.X
        
        [M,N] = data.shape
        #first, substract the mean of each dimension (=sample type)
        if shiftData:
            data_mean = np.mean(data,1)
            data = data - np.tile(data_mean, (N,1)).T
            self.X_shifted = data
        #calculate the coveriance matrix
        self.data_covariance = (1 / (N-1)) * (np.matmul(data , data.T))
        #find the eigenvalues and eigenvectors
        #(using eigh function which assumes a real and symettric matrix)
        [self.v,self.PCs] = np.linalg.eigh(self.data_covariance)
        #normalizing the eigenvmaxectors to be in precntage
        self.v = self.v / np.sum(self.v)
        #sort the values in decreasing order
        sorted_indexexs = np.argsort(-self.v) #get the indexes first
        self.v   = self.v[sorted_indexexs]
        self.PCs[:] = self.PCs[:,sorted_indexexs] 
        #project the data set on the PCs space
        self.samples_PC_space = np.matmul(self.PCs.T , data)
        print('Finished computing PCA\n========================\n')
        print('Variance of the first 5 principal components:\n')
        print(self.v[0:4])