#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:57:38 2019

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

#Fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)


#Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg =  PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)


#Visualizing Linear Regression results
plt.scatter(X,y,color='red')
plt.plot(X, lin_reg.predict(X),color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#Visualizing Polynomial Regression results

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y,color='red')
#plt.plot(X, lin_reg2.predict(X_poly),color='blue')
#plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or Bluff (Polynomial Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



#Visualizing both Linear & Polynomial Regression results in same graph
plt.scatter(X,y,color='red')
plt.plot(X, lin_reg.predict(X),color='black')
plt.plot(X, lin_reg2.predict(X_poly),color='blue')
plt.title('Truth or Bluff (Polynomial Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Predicting a new result with Linear Regression

lin_reg.predict([[6.5]])


# Predicting a new result with Linear Regression

lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
























