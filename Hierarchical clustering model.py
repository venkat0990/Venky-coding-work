# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 14:49:35 2019

@author: Don't Give up
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# Using dendrogram to obtain optimum number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting hierarchichal clustering to the mall dataset
from sklearn. cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', 
                             linkage = 'ward')
Y_hc = hc.fit_predict(X)

# Visualising the hierarchical clustering results
plt.scatter(X[Y_hc == 0, 0], X[Y_hc == 0, 1], s = 100, c = 'pink', 
            label = 'Careful')
plt.scatter(X[Y_hc == 1, 0], X[Y_hc == 1, 1], s = 100, c = 'orange', 
            label = 'Standard')
plt.scatter(X[Y_hc == 2, 0], X[Y_hc == 2, 1], s = 100, c = 'purple', 
            label = 'Target')
plt.scatter(X[Y_hc == 3, 0], X[Y_hc == 3, 1], s = 100, c = 'grey', 
            label = 'Careless')
plt.scatter(X[Y_hc == 4, 0], X[Y_hc == 4, 1], s = 100, c = 'maroon', 
            label = 'Sensible')
plt.title('Clusters of clients')
plt.xlabel('Annual income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()