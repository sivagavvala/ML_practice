#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 08:01:21 2019

@author: siva
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 1].values


#from sklearn.preprocessing import Imputer
#imputer= Imputer(missing_values='NaN',strategy='mean',axis=0)
#imputer.fit(x[:,1:3])
#x[:,1:3]= imputer.transform(x[:,1:3])


#Splitting test and train data
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size = 1/3,random_state=0)


#Feature Scaling 
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""

#Fitting Simple linear regression to training set
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x_train,y_train)

# Predicting Test results
y_pred = linear_reg.predict(x_test)


# Visualizing Training set results
plt.scatter(x_train,y_train, color='red')
plt.plot(x_train, linear_reg.predict(x_train),color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Yrs of Experience')
plt.ylabel('Salary')
plt.show()




# Visualizing Test set results
plt.scatter(x_test,y_test, color='red')
plt.plot(x_train, linear_reg.predict(x_train),color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Yrs of Experience')
plt.ylabel('Salary')
plt.show()

































     




















