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
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

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
        if typ==1: #I(V) curves
            
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
            
            
        if typ==2: #force-distance curves
            
            # a function to find the index of the element in an array with the nearest value to a value
            def find_nearest(array, value):
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                return idx
            
            n=0
            heads = ["Vertical Tip Position [m]", "Vertical Deflection [N]", "Height [m]", "Error Signal [V]", "Head Height [m]", "Head Height (measured & smoothed) [m]", "Head Height (measured) [m]", "Height (measured & smoothed) [m]", "Height (measured) [m]", "Lateral Deflection [m]", "Series Time [m]", "Segment Time [m]"]
            self.X_force    = []
            self.X_position = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if os.path.splitext(file)[1] == '.txt':
                        print('Working On: \n')
                        print(os.path.join(root,file) + '\n')
                        files_legnth = str(len(files))
                        print(str(n+1) + ' Out of: ' + files_legnth)
                        df = pd.read_csv(os.path.join(root,file) , skiprows=16, names=heads, sep=' ')
                        self.X_force.append(np.array(df["Vertical Deflection [N]"]))
                        self.X_position.append(np.array(df["Head Height (measured & smoothed) [m]"]))
                    n+=1
            #convert the list to a numpy array
            self.X_force = np.transpose(np.array(self.X_force))
            self.X_position = np.transpose(np.array(self.X_position))
            
            #Callibrating the position such that all the samples will "talk in the same language"
            self.x_modified = np.linspace(0 , 200e-9, 851)
            
            sns.set_palette(sns.color_palette("coolwarm", 1733), 1733)
            self.X_position_modified = copy.deepcopy(self.X_position) #use deep copy to create a separate calss instance
            self.X_ofInterest = []
            j=1
            self.X_force.tolist()
            for column in self.X_position_modified.T:
                print('Callibrating sample number ' + str(j) + ' Out of: ' + files_legnth)
                f = interp1d(column[0:1100], column[0:1100]) #linear interpolate the position upon itself
                for i in self.x_modified: 
                    interp_value = f(i) #check the interpolated value
                    idx = find_nearest(column, interp_value) #get the index of the neasrest value to the common position
                    column[idx] = interp_value #replace the value of the nearest value to the interp value
                idx = find_nearest(column, 0)
                self.X_ofInterest.append(self.X_force[idx : (idx+851) , j-1 ])
                j+=1
            #convert the list to a numpy array
            self.X_ofInterest = np.transpose(np.array(self.X_ofInterest))
            self.X_force = np.array(self.X_force)
            
            print('New data_2_cluster object created\n=================================\n')
            print('Object dimensions:\n')
            print('Number of sample types: ' + str(self.X_force.shape[0])+'\n')
            print('Number of samples taken: ' + str(self.X_force.shape[1])+'\n')
                
    def plot(self):
        #This function plots the object attributes according to the user selection
        
        def plot_pca(xd, PC_count, PCs, samples_PC_space, v, xlabel, ylabel):
            ax1 = plt.subplot2grid((2,1), (0,0))
            ax2 = plt.subplot2grid((2,1), (1,0))
                
            for n in range(PC_count):
                ax1.plot(xd, PCs[:,n], label=n)
                ax1.set_xlabel(xlabel)
                ax1.set_ylabel(ylabel)
                ax1.legend(loc=0)
                ax2.scatter(samples_PC_space[0,:], samples_PC_space[1,:] )
                ax2.set_xlabel('PC #1 (' + str(round(v[0]*100,2)) + ' %)')
                ax2.set_ylabel('PC #2 (' + str(round(v[1]*100,2)) + ' %)')
                ax1.set_title('PCA Results - ' + self.name + ' data') 
                plt.tight_layout()
        
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
            f = plt.figure(4)
            print('Plotting the histograms of the data')
            ax1 = plt.subplot2grid((1,3), (0,0))
            ax2 = plt.subplot2grid((1,3), (0,1), sharey=ax1)
            ax3 = plt.subplot2grid((1,3), (0,2), sharey=ax1)
            for column in self.X_norm.T:
                ax1.plot(self.voltages_a, column)
                ax1.set_xlabel('Voltage [V]', fontsize=12)
                ax1.set_ylabel('Normalized Current', fontsize=12)
            for column in self.X_hist.T:
                ax2.plot(column, self.bins, 'o--')
                ax2.set_xlabel('Probability Density', fontsize=12)
                ax2.set_yticks([])
            for column in self.X_KDE.T:
                ax3.plot(column, self.bins_KDE, '--')
                ax3.set_xlabel('Probability Density', fontsize=12)
                ax3.set_yticks([])
            f.suptitle('Data Histograms & KDE - ' + self.name, y=0.98)
            f.subplots_adjust(wspace=None, hspace=None, top=0.92)
        if 'PCA' in selections:
            typ = input('Which PCA would you like to plot?\nChoose from: data, data_norm, data_histograms, data_KDE\n')
            typ = typ.split(',')
            PC_count = input('How many PCs would you like to plot?\n')
            PC_count = int(PC_count)
            plt.figure(5)
            if 'data' in typ: 
                plot_pca(self.voltages_a, PC_count, self.X_PCs, self.X_samples_PC_space, self.X_v, 'Voltage [V]', 'Current')
            if 'data_norm' in typ:
                plot_pca(self.voltages_a, PC_count, self.X_norm_PCs, self.X_norm_samples_PC_space, self.X_norm_v, 'Voltage [V]', 'Current')
            if 'data_histograms' in typ:
                plot_pca(self.bins, PC_count, self.X_hist_PCs, self.X_hist_samples_PC_space, self.X_hist_v, 'Normalized Current', 'Probability Density')
            if 'data_KDE' in typ:
                plot_pca(self.bins_KDE, PC_count, self.X_KDE_PCs, self.X_KDE_samples_PC_space, self.X_KDE_v, 'Normalized Current', 'Probability Density')
    '''        
    def create_diff(self, norm=True):
        #this function calculates the numerical derivative of each curve (column) and creates a new matrix
        
        def diff(x,y, frc=0.3, itt=2 ):
            #smoothing the data before doing a derivative
            data_smoothed = lowess(y,x,  is_sorted=True, frac=frc, it=itt)
            dydx = np.diff(data_smoothed[:,1])/np.diff(data_smoothed[:,0]) #numerical derivative
            x = np.asarray(x)
            xd = (x[1:]+x[:-1])/2 #the derivative gives the value at the midpoints
            filtered = lowess(dydx,xd,  is_sorted=True, frac=0.05, it=2)
            #smooth the data:
            f = interp1d(xd, dydx, kind='cubic')
            smooth_xnew = np.linspace(xd[0], xd[len(xd)-1], 4*len(xd))
            return smooth_xnew, f(smooth_xnew), filtered[:,0], filtered[:,1], xd, dydx
                
        if norm:
            data = self.X_norm
        else:
        #    data = self.X
        
        #data_diff = []
        #for column in data.T:
            #calculate the derivative for each curve (column)
            #smooth_xnew, f(smooth_xnew), filtered[:,0], filtered[:,1], xd, dydx = diff(self.voltages_a, data, frc=0.1, itt=2)
            #data_diff.append(filtered[:,1])
        
        data_diff = np.transpose(np.array(data_diff)) #convert to a numpy array
        
        if norm:
            self.X_norm_diff = data_diff
        else:
            self.X = data_diff
        
        #self.voltages_diff = filtered[:,0]                                                      
       '''
    def pca(self, shiftData=False):
        #this function calculaation the principal components of the data
        
        typ = input('Which data would you like to calculate PCA for?\nChoose from: data,data_norm,data_histograms,data_KDE\n')
        
        if typ == 'data_norm':
            data = self.X_norm
        if typ == 'data':
            data = self.X
        if typ == 'data_histograms':
            data = self.X_hist
        if typ == 'data_KDE':
            data =self.X_KDE
        
        [M,N] = data.shape
        #If needed, substract the mean of each dimension (=sample type)
        if shiftData:
            data_mean = np.mean(data,1)
            data = data - np.tile(data_mean, (N,1)).T
            self.X_shifted = data
        #calculate the coveriance/correlation matrix
        # In order to avoid zero standard deviation check before if there are rows of zero
        if True in (data == 0).all(axis=1):
            data_covariance = np.cov(data)
            print('Used covariance matrix\n')
        else: #if there are not rows of zero, than there will be no dividing by zero
            data_covariance = np.corrcoef(data)
            print('Used correlation matrix\n')
        #find the eigenvalues and eigenvectors
        #(using np.linalg.eigh function which assumes a real and symettric matrix)
        [v,PCs] = np.linalg.eigh(data_covariance)
        #normalizing the eigenvmaxectors to be in precntage
        v = v / np.sum(v)
        #sort the values in decreasing order
        sorted_indexexs = np.argsort(-v) #get the indexes first
        v   = v[sorted_indexexs]
        PCs[:] = PCs[:,sorted_indexexs] 
        #project the data set on the PCs space
        samples_PC_space = np.matmul(PCs.T , data)
        if typ == 'data_norm':
            self.X_norm_covariance       = data_covariance
            self.X_norm_v                = v
            self.X_norm_PCs              = PCs
            self.X_norm_samples_PC_space = samples_PC_space
        if typ == 'data':
            self.X_covariance       = data_covariance
            self.X_v                = v
            self.X_PCs              = PCs
            self.X_samples_PC_space = samples_PC_space
        if typ == 'data_histograms':
            self.X_hist_covariance       = data_covariance
            self.X_hist_v                = v
            self.X_hist_PCs              = PCs
            self.X_hist_samples_PC_space = samples_PC_space
        if typ == 'data_KDE':
            self.X_KDE_covariance       = data_covariance
            self.X_KDE_v                = v
            self.X_KDE_PCs              = PCs
            self.X_KDE_samples_PC_space = samples_PC_space
        print('Finished computing PCA\n========================\n')
        print('Variance of the first 5 principal components:\n')
        print(v[0:4])
    
    def create_hist(self, norm=True):
        #This function runs over each curve (column in the data)
        #and computes a histogram
        
        if norm:
            data = self.X_norm
        else:
            data = self.X
        
        #first, create a vector with the sum of all histograms
        #then, use the bins for the indicidual curves
        self.X_allHistograms, bns = np.histogram(data, bins='auto', density=True)
        self.bins_KDE = np.linspace(np.min(data), np.max(data), len(bns)*2)
        #Calculate the histograms and KDE
        
        X_hist = []
        X_KDE  = []
        for column in data.T:
            hist, bin_edges = np.histogram(column, bins=bns, density=True) #the number of bins is the legnth of the vector divided by bns
            X_hist.append(hist)
        
            #Calculate the kernel density estimation
            # instantiate and fit the KDE model
            bndwdth = (np.max(data)-np.min(data))/(150)
            kde = KernelDensity(bandwidth=bndwdth, kernel='gaussian')
            kde.fit(column[:, None])
            # score_samples returns the log of the probability density
            logprob = kde.score_samples(self.bins_KDE[:, None])
            prob = np.exp(logprob)
            X_KDE.append(prob)
            
        self.bins   = bin_edges[:-1] + np.diff(bin_edges) / 2 #get the bins centers for plotting
        self.X_hist = np.transpose(np.array(X_hist)) #convert to a numpy array
        self.X_KDE  = np.transpose(np.array(X_KDE)) #convert to a numpy array
        
        print('Finished computing the histogram and KDE\n')
        print('Range:')
        print([np.min(data),np.max(data)])
        

class merge_data(object):
    def __init__(self, data_sets):
        
        self.X_merged=False
        self.X_norm_merged=False
        former_X_set = data_sets[0].X
        former_Xnorm_set = data_sets[0].X_norm
        for data_set in data_sets:
            if self.X_merged == False:
                self.X_merged = former_X_set
                self.X_norm_merged = former_Xnorm_set
            else:
                self.X_merged = np.append(self.X_merged, data_set.X)
                self.X_norm_merged = np.append(self.X_norm_merged, data_set.X_norm)