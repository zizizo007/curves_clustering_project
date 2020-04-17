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

import seaborn as sns
sns.set_context("talk")
sns.set_style("white")

class data_2_cluster(object):
    
    def __init__(self, path, typ=1):
        # when initializing the main object:
        # X,X_norm - the data and the normlized data (according to z-score) are added as an array
        # X_df,X_norm_df - the data and the normlized data are added as a dataframe
        
        #Define a name for the data set (user can override)
        self.name = ''
        
        #create empty arrays for the loaded data
        self.X      = []
        self.X_norm = []
        
        #first, go over the source folder and load the data into a matrix X
        if typ==1:
            
            self.voltages_a = []    
            for voltage in np.arange(-2,2.01,0.01):
                self.voltages_a.append(round(voltage,2))
            
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
        selections = input('What would you like to plot?\nChoose from: X,X_norm,X_shifted,X_histograms,PCA\n')
        selections = selections.split(',')
        if 'X' in selections:
            plt.figure(1)
            print('Plotting the matrix X:\n')
            for column in self.X.T:
                plt.plot(self.voltages_a,column)
            plt.title('Data Curves (Raw) - ' + self.name)
            plt.xlabel('Voltage [V]')
            plt.ylabel('Current [A]')
            plt.tight_layout()
        if 'X_norm' in selections:
            plt.figure(2)
            print('Plotting the normalized matrix X:\n')
            for column in self.X_norm.T:
                plt.plot(self.voltages_a,column)
            plt.title('Data Curves (normalized) - ' + self.name)
            plt.xlabel('Voltage [V]')
            plt.ylabel('Normalized Current')
            plt.tight_layout()
        if 'X_shifted' in selections:
            plt.figure(3)
            print('Plotting the normalized shifted matrix X:\n')
            for column in self.X_shifted.T:
                plt.plot(self.voltages_a,column)
            plt.title('Data Shifted to Zero Means - ' + self.name)
        if 'X_histograms' in selections:
            plt.figure(4)
            print('Plotting the histograms of the data')
            ax1 = plt.subplot2grid((1,2), (0,0))
            ax2 = plt.subplot2grid((1,2), (0,1), sharey=ax1)
            for column in self.X_norm.T:
                ax1.plot(self.voltages_a, column)
                ax1.set_xlabel('Voltage [V]')
                ax1.set_ylabel('Normalized Current')
            for column in self.X_hist.T:
                ax2.plot(column, self.bins, 'o--')
                ax1.set_xlabel('Counts')
            plt.title('Data Histograms - ' + self.name)
            plt.tight_layout()
        if 'PCA' in selections:
            typ = input('Which PCA would you like to plot?\nChoose from: data, data_histograms\n')
            typ = typ.split(',')
            PC_count = input('How many PCs would you like to plot?\n')
            PC_count = int(PC_count)
            plt.figure(5)
            if 'data' in typ: 
                ax1 = plt.subplot2grid((2,1), (0,0))
                ax2 = plt.subplot2grid((2,1), (1,0))
                
                for n in range(PC_count):
                    ax1.plot(self.voltages_a, self.PCs[:,n], label=n)
                ax1.set_xlabel('Voltage [V]')
                ax1.set_ylabel('Current [?]')
                ax1.legend(loc=0)
                ax2.scatter(self.samples_PC_space[0,:], self.samples_PC_space[1,:] )
                ax2.set_xlabel('PC #1 (' + str(round(self.v[0]*100,2)) + ' %)')
                ax2.set_ylabel('PC #2 (' + str(round(self.v[1]*100,2)) + ' %)')
                ax1.set_title('PCA Results - ' + self.name + ' data') 
                plt.tight_layout()     
            if 'data_histograms' in typ:
                ax1 = plt.subplot2grid((2,1), (0,0))
                ax2 = plt.subplot2grid((2,1), (1,0))
                
                for n in range(PC_count):
                    ax1.plot(self.bins, self.PCs[:,n], label=n)
                ax1.set_xlabel('Bin #')
                ax1.set_ylabel('[?]')
                ax1.legend(loc=0)
                ax2.scatter(self.samples_PC_space[0,:], self.samples_PC_space[1,:] )
                ax2.set_xlabel('PC #1 (' + str(round(self.v[0]*100,2)) + ' %)')
                ax2.set_ylabel('PC #2 (' + str(round(self.v[1]*100,2)) + ' %)')
                ax1.set_title('PCA Results - ' + self.name + ' data_histograms') 
                plt.tight_layout()     
                
                               
    def pca(self, norm=True, shiftData=False):
        #this function calculaation the principal components of the data
        
        typ = input('Which data would you like to calculate PCA for?\nChoose from: data,data_norm,data_histograms\n')
        
        if typ == 'data_norm':
            data = self.X_norm
        if typ == 'data':
            data = self.X
        if typ == 'data_histograms':
            data = self.X_hist
        
        [M,N] = data.shape
        #If needed, substract the mean of each dimension (=sample type)
        if shiftData:
            data_mean = np.mean(data,1)
            data = data - np.tile(data_mean, (N,1)).T
            self.X_shifted = data
        #calculate the coveriance/correlation matrix
        # In order to avoid zero standard deviation check before if there are rows of zero
        if True in (data == 0).all(axis=1):
            self.data_covariance = np.cov(data)
        else: #if there are not rows of zero, than there will be no dividing by zero
            self.data_covariance = np.corrcoef(data)
        #find the eigenvalues and eigenvectors
        #(using np.linalg.eigh function which assumes a real and symettric matrix)
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
    
    def create_hist(self, norm=True):
        #This function runs over each curve (column in the data)
        #and computes a histogram
        
        if norm:
            data = self.X_norm
        else:
            data = self.X
        
        #first, create a vector with the sum of all histograms
        #then, use the bins for the indicidual curves
        self.X_allHistograms, bns = np.histogram(data, bins='auto')
        
        #Calculate the histograms
        X_hist = []
        for column in data.T:
            hist, bin_edges = np.histogram(column, bins=bns) #the number of bins is the legnth of the vector divided by bns
            X_hist.append(hist)
        self.bins = bin_edges[:-1] + np.diff(bin_edges) / 2 #get the bins centers for plotting
        self.X_hist = np.transpose(np.array(X_hist)) #convert to a numpy array
        
        print('Finished computing the histogram\n')
        print('Bins range:')
        print([np.min(data),np.max(data)])