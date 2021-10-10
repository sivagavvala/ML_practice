#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 08:54:54 2019

@author: siva
"""


#%reset -f

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values


#Using Elbow method finding optimal no of Clusters
from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init='k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('No of cluseters')
plt.ylabel('WCSS')
plt.show()


#Applying K Means to thre mall dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#y-kmeans says which record belongs to which cluster


#Visualizing the clusters  (This is only for 2Dimensions)
plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1], s = 100, c = 'red', label = 'customer 1')
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1], s = 100, c = 'blue', label = 'customer 2')
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1], s = 100, c = 'green', label = 'customer 3')
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1], s = 100, c = 'cyan', label = 'customer 4')
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4,1], s = 100, c = 'magenta', label = 'customer 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (K$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()





























    



