# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 3].values


#from sklearn.preprocessing import Imputer
#imputer= Imputer(missing_values='NaN',strategy='mean',axis=0)
#imputer.fit(x[:,1:3])
#x[:,1:3]= imputer.transform(x[:,1:3])


# Take cate missing data
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(x[:, 1:3])
x[:, 1:3]=missingvalues.transform(x[:, 1:3])

#-------    missingvalues.statistics_

pd.DataFrame(x,x[:,1:3]).head()

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder()
x[:, 0]=labelencoder_x.fit_transform(x[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray()

#Encoding the dependent variable
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)



#Splitting test and train data
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size = 0.2,random_state=0)


#Feature Scaling 
from sklearn.preprocessing import StandardScaler


sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)


display(x_test)

















