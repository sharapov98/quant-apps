import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

#import the feature and response datafiles
feature = pd.read_csv("xin.csv",
                     sep=",")
response = pd.read_csv("yin.csv",
                      sep=",")

#preprocess data: convert dataframe to array
feature=np.array(feature)
response=np.array(response)

from scipy.stats import norm
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import math

class GKR:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    '''Calculate optimal bandwith'''
    def optimal_band(self, x):
        params = {'bandwidth': np.logspace(-1, 1, 10)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(x)
        return format(grid.best_estimator_.bandwidth)
    
    '''Implement the Gaussian Kernel'''
    def gaussian_kernel(self, z):
        return (1/math.sqrt(2*math.pi))*math.exp(-0.5*z**2)
    
    '''Calculate weights and return prediction'''
    def predict(self, X):
        kernels = [self.gaussian_kernel((xi-X)/self.optimal_band) for xi in self.x]
        weights = [len(self.x) * (kernel/np.sum(kernels)) for kernel in kernels]
        return np.dot(weights, self.y)/len(self.x)
 


import pylab as pl
pl.clf()
pl.plot(x, y, label='y noisy')
pl.plot(x, yest, label='y pred')
pl.legend()
pl.show()