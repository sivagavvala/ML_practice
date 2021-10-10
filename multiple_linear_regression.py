#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:30:37 2019

@author: siva
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:,4 ].values



#from sklearn.preprocessing import Imputer
#imputer= Imputer(missing_values='NaN',strategy='mean',axis=0)
#imputer.fit(x[:,1:3])
#x[:,1:3]= imputer.transform(x[:,1:3])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
X[:, 3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()


#Avoiding Dummy variable Trap

X=X[:,1:]


display(X)

#Splitting test and train data
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X, y,test_size = 0.2,random_state=0)




#Feature Scaling 
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""

#Fitting Simple linear regression to training set
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X_train,y_train)

# Predicting Test results
y_pred = linear_reg.predict(X_test)


#Biulding optimal model using backward elimaination
import statsmodels.api as sm

#X=np.append(arr = X,values=np.ones(50,1)).astype(int),ax   is=1)
X=np.append(arr = np.ones((50,1)).astype(int),values=X,axis=1)

#Ordinary Least Squares (OLS)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

  

X_opt=X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()


X_opt=X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

#Check Adj R Squares value here and assess along with p-value
X_opt=X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()


#Automation of Backward elimination

import statsmodels.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

regressor_OLS.summary()




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


