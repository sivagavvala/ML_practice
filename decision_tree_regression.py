#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 05:35:56 2019

@author: siva
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:, 1:2].values
y=dataset.iloc[:, 2].values


#from sklearn.preprocessing import Imputer
#imputer= Imputer(missing_values='NaN',strategy='mean',axis=0)
#imputer.fit(x[:,1:3])
#x[:,1:3]= imputer.transform(x[:,1:3])

'''
#Splitting test and train data
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size = 0.2,random_state=0)
'''

'''
#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
'''

#Fitting the Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

# Predicting a new result with Linear Regression
y_pred = regressor.predict(([[6.5]]))


#Visualizing Decision Tree Regression Regression results
plt.scatter(X,y,color='red')
plt.plot(X, regressor.predict(X),color='blue')
plt.title('Truth or Bluff (Decision Tree Regression Modelf)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#Visualizing Polynomial Regression results for higher resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X,y,color='red')
plt.plot(X_grid, regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff (Regression Modelf)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()









