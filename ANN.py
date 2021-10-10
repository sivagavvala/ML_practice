#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 19:52:41 2019

@author: siva
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:, 3:13].values
y=dataset.iloc[:, 13].values



#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1=LabelEncoder()
X[:, 1]=labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2=LabelEncoder()
X[:, 2]=labelencoder_X_2.fit_transform(X[:, 2])


onehotencoder = OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]


#Splitting test and train data
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size = 0.2,random_state=0)


#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#PART 2

#Importing Keras Libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer to first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
#output dim is avg of input and output layers


# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the training
classifier.fit(X_train, y_train, batch_size = 20, epochs = 100)


# Predicting test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)




# Making the predictions and evaluating the model
    























 






