#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 08:13:49 2019

@author: siva
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values


#using dendogram to find optimal no of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward')) #ward minimises variance
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Dist')
plt.show()

#Fitting Hierarchial Clustering to data set
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)



#Visualizing the clusters  (This is only for 2Dimensions)
plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1], s = 100, c = 'red', label = 'careful')
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1], s = 100, c = 'blue', label = 'standard')
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1], s = 100, c = 'green', label = 'target')
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1], s = 100, c = 'cyan', label = 'careless')
plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1], s = 100, c = 'magenta', label = 'sensible')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (K$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()














