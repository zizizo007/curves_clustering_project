# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import linregress
import scipy.signal as sig
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
                arr_diff = (np.abs(array - value))
                idx = arr_diff.argmin()
                err = np.amin(arr_diff)
                return idx, err
            
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
            steps = 550
            self.x_modified = np.linspace(0 , 200e-9, steps)
            
            
            self.X_position_modified = [] 
            self.X_ofInterest = []
            self.X_position_diff = []
            j=1
            self.X_force.tolist()
            for column in self.X_position.T: # for each measurment
                print('Callibrating sample number ' + str(j) + ' Out of: ' + files_legnth)
               
                idx_0, err_0 = find_nearest(column, 0) #get the index of the neasrest value to zero
                self.X_ofInterest.append(self.X_force[idx_0 : (idx_0+steps) , j-1 ])
                self.X_position_modified.append(column[idx_0 : (idx_0+steps)])
                self.X_position_diff.append( np.diff(column[idx_0 : idx_0+steps]) ) 
                
                j+=1
            #convert the list to a numpy array
            self.X_ofInterest = np.transpose(np.array(self.X_ofInterest))
            self.X_position_modified = np.transpose(np.array(self.X_position_modified))
            self.X_position_diff = np.transpose(np.array(self.X_position_diff))
            self.X_force = np.array(self.X_force)
            
            #set the common positions according to the mean of the diffrences
            self.x_modified = np.zeros(steps)
            self.x_modified[0] = 0
            prev = 0
            for k in range(1,steps):
                self.x_modified[k] = prev + np.mean(self.X_position_diff[k-1,:])
                prev = self.x_modified[k]
            
            #compute the error of the common positions, add update them accordingly
            self.x_modified_err = np.abs(self.X_position_modified - self.x_modified.reshape( len(self.x_modified),1) )
            err_mean = np.mean(self.x_modified_err, axis=1)
            self.x_modified = self.x_modified - err_mean
            
            print('New data_2_cluster object created\n=================================\n')
            print('Object dimensions:\n')
            print('Number of sample types: ' + str(self.X_force.shape[0])+'\n')
            print('Number of samples taken: ' + str(self.X_force.shape[1])+'\n')
                
    def plot(self):
        #This function plots the object attributes according to the user selection
        
        def setupPalette(count, pal=None):
            # See http://xkcd.com/color/rgb/. These were chosen to be different "enough".
            colors = ['windows blue' , 'amber' , 'faded green' , 'dusty purple' , 'pale red',
                      'grass green', 'dirty pink', 'azure', 'tangerine', 'strawberry',
                      'yellowish green', 'gold', 'sea blue', 'lavender', 'orange brown', 'turquoise',
                      'royal blue', 'cranberry', 'pea green', 'vermillion', 'sandy yellow', 'greyish brown',
                      'magenta', 'silver', 'ivory', 'carolina blue', 'very light brown']
        
            palette = sns.color_palette(palette=pal, n_colors=count) if pal else sns.xkcd_palette(colors)
            sns.set_palette(palette, n_colors=count)

        setupPalette(20)
        
        def plot_pca(xd, PC_count, PCs, samples_PC_space, v, xlabel, ylabel):
            ax1 = plt.subplot2grid((2,1), (0,0))
            ax2 = plt.subplot2grid((2,1), (1,0))
                
            for n in range(PC_count):
                ax1.plot(xd, PCs[:,n], label='PC #' + str(n+1))
                ax1.set_xlabel(xlabel)
                ax1.set_ylabel(ylabel)
                ax1.legend(loc='upper right')
                ax1.set_xlim([xd[0],xd[-1]])
                ax2.scatter(samples_PC_space[0,:], samples_PC_space[1,:] , edgecolors='k', alpha=0.2)
                ax2.set_xlabel('PC #1 (' + str(round(v[0]*100,2)) + ' %)')
                ax2.set_ylabel('PC #2 (' + str(round(v[1]*100,2)) + ' %)')
                ax2.set_xlim([np.min(samples_PC_space[0,:]) , np.max(samples_PC_space[0,:])])
                ax2.set_ylim([np.min(samples_PC_space[1,:]) , np.max(samples_PC_space[1,:])])
                ax1.set_title('PCA Results - ' + self.name + ' data') 
                plt.tight_layout()
        
        selections = input('What would you like to plot?\nChoose from: X,X_norm,X_shifted,X_histograms,force_Curves,PCA\n')
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
        if 'force_Curves' in selections:
            [M,N] = self.X_ofInterest.shape
            sns.set_palette(sns.color_palette("coolwarm", N), N)
            for column in self.X_ofInterest.T:
                plt.plot(self.x_modified, column, lw=0.4)
            plt.xlim(self.x_modified[0],self.x_modified[-1])
            plt.title('All Force Curves')
        if 'PCA' in selections:
            typ = input('Which PCA would you like to plot?\nChoose from: data, data_norm, data_histograms, data_KDE, data_AFM\n')
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
            if 'data_AFM' in typ:
                plot_pca(self.x_modified,PC_count, self.X_AFM_PCs, self.X_AFM_samples_PC_space, self.X_AFM_v, 'Position [m]', 'Force')
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
        
        typ = input('Which data would you like to calculate PCA for?\nChoose from: data,data_norm,data_histograms,data_KDE,data_forceCurves\n')
        
        if typ == 'data_norm':
            data = self.X_norm
        if typ == 'data':
            data = self.X
        if typ == 'data_histograms':
            data = self.X_hist
        if typ == 'data_KDE':
            data =self.X_KDE
        if typ == 'data_forceCurves':
            data = self.X_ofInterest
        
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
        if typ == 'data_forceCurves':
            data_covariance = np.cov(data)
            print('Used covariance matrix\n')
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
        if typ == 'data_forceCurves':
            self.X_AFM_covariance       = data_covariance
            self.X_AFM_v                = v
            self.X_AFM_PCs              = PCs
            self.X_AFM_samples_PC_space = samples_PC_space
        
        print('Finished computing PCA\n========================\n')
        print('Variance of the first 5 principal components:\n')
        print(v[0:5])
    
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
        typ = input('Which data type would you like to merge?\nChoose from: data,data_norm,data_histograms,data_KDE,data_AFM\n')
        
        #merge the data into a single matrix by using the append function
        sets_lengths = []
        if typ == 'data_AFM':
            self.X_AFM_merged = []
            former_X_AFM = data_sets[0].X_ofInterest
            for data_set in data_sets:
                sets_lengths.append(data_set.X_ofInterest.shape[1])
                if len(self.X_AFM_merged) == 0:
                    self.X_AFM_merged = former_X_AFM
                else:
                    self.X_AFM_merged = np.append(self.X_AFM_merged, data_set.X_ofInterest, axis=1)
        
        #get the indexes of the merged data sets
        for i in range(len(sets_lengths)):
            if i==0:
                sets_lengths[i] = np.sum(sets_lengths[0:1])
            else:    
                sets_lengths[i] = np.sum(sets_lengths[i-1:i+1])
        self.datas_length = sets_lengths
        
        #save the object in an array as the new object attribute
        self.all_objects = data_sets
        
    def plot(self):
        #This function plots the object attributes according to the user selection
        def setupPalette(count, pal=None):
            # See http://xkcd.com/color/rgb/. These were chosen to be different "enough".
            colors = ['windows blue' , 'amber' , 'faded green' , 'dusty purple' , 'pale red',
                      'grass green', 'dirty pink', 'azure', 'tangerine', 'strawberry',
                      'yellowish green', 'gold', 'sea blue', 'lavender', 'orange brown', 'turquoise',
                      'royal blue', 'cranberry', 'pea green', 'vermillion', 'sandy yellow', 'greyish brown',
                      'magenta', 'silver', 'ivory', 'carolina blue', 'very light brown']
        
            palette = sns.color_palette(palette=pal, n_colors=count) if pal else sns.xkcd_palette(colors)
            sns.set_palette(palette, n_colors=count)

        setupPalette(20)
        
        def plot_pca(xd, PC_count, PCs, samples_PC_space, v, xlabel, ylabel, datas_lengths):
            ax1 = plt.subplot2grid((2,1), (0,0))
            ax2 = plt.subplot2grid((2,1), (1,0))
                
            for n in range(PC_count):
                ax1.plot(xd, PCs[:,n], label='PC #' + str(n+1), color='C'+str(n + len(datas_lengths)))
                ax1.set_xlabel(xlabel)
                ax1.set_ylabel(ylabel)
                ax1.legend(loc='upper right')
                ax1.set_xlim([xd[0],xd[-1]])
            current_data = 0
            for length in range(len(datas_lengths)):
                if length == 0:
                    ax2.scatter(samples_PC_space[0,0:datas_lengths[0]], samples_PC_space[1,0:datas_lengths[0]] , edgecolors='k', alpha=0.1, color='C'+str(current_data ))
                    print(str(0) + ',')
                    print(str(datas_lengths[0]))
                else:
                    ax2.scatter(samples_PC_space[0,datas_lengths[length-1]:datas_lengths[length]], samples_PC_space[1,datas_lengths[length-1]:datas_lengths[length]] , edgecolors='k', alpha=0.2, color='C'+str(current_data ))
                    print(str(datas_lengths[length-1]) + ',')
                    print(str(datas_lengths[length]))
                current_data+=1
            ax2.set_xlabel('PC #1 (' + str(round(v[0]*100,2)) + ' %)')
            ax2.set_ylabel('PC #2 (' + str(round(v[1]*100,2)) + ' %)')
            ax2.set_xlim([np.min(samples_PC_space[0,:]) , np.max(samples_PC_space[0,:])])
            ax2.set_ylim([np.min(samples_PC_space[1,:]) , np.max(samples_PC_space[1,:])])
            ax1.set_title('PCA Results - ' + self.name + ' data') 
            plt.tight_layout()
        
        selections = input('What would you like to plot?\nChoose from: X,X_norm,X_shifted,X_histograms,force_Curves,PCA\n')
        selections = selections.split(',')
        
        
        if 'force_Curves' in selections:
            n=0
            current_set = 0
            for column in self.X_AFM_merged.T:
                if n in self.datas_length: 
                #this condition flags in which set we are
                    if n==0:
                        current_set=0
                    else:
                        current_set+=1
                clr = 'C'+str(current_set)
                plt.plot(self.all_objects[current_set].x_modified, column, lw=0.4, color=clr)
                n+=1
            plt.xlim(self.all_objects[0].x_modified[0],self.all_objects[0].x_modified[-1])
            plt.title('All Force Curves')
        if 'PCA' in selections:
            typ = input('Which PCA would you like to plot?\nChoose from: data, data_norm, data_histograms, data_KDE, data_AFM\n')
            typ = typ.split(',')
            PC_count = input('How many PCs would you like to plot?\n')
            PC_count = int(PC_count)
            plt.figure(5)
            if 'data_AFM' in typ:
                plot_pca(self.all_objects[0].x_modified,PC_count, self.X_AFM_PCs, self.X_AFM_samples_PC_space, self.X_AFM_v, 'Position [m]', 'Force', self.datas_length)
        

    
    def pca(self, shiftData=False):
        #this function calculaation the principal components of the data
        
        typ = input('Which data would you like to calculate PCA for?\nChoose from: data,data_norm,data_histograms,data_KDE,data_forceCurves\n')
        
        if typ == 'data_norm':
            data = self.X_norm
        if typ == 'data':
            data = self.X
        if typ == 'data_histograms':
            data = self.X_hist
        if typ == 'data_KDE':
            data =self.X_KDE
        if typ == 'data_forceCurves':
            data = self.X_AFM_merged
        
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
        if typ == 'data_forceCurves':
            data_covariance = np.cov(data)
            print('Used covariance matrix\n')
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
        if typ == 'data_forceCurves':
            self.X_AFM_covariance       = data_covariance
            self.X_AFM_v                = v
            self.X_AFM_PCs              = PCs
            self.X_AFM_samples_PC_space = samples_PC_space
        
        print('Finished computing PCA\n========================\n')
        print('Variance of the first 5 principal components:\n')
        print(v[0:5])