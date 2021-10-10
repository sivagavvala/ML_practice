#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:17:07 2019

@author: siva
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t' , quoting = 3) 
#by including quoting 3 ignoring double quotes


#Clearing the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    #Stemming is done to avoid sparsity
    #will lead to single word that helps to classify o/p
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating Bag of words model
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Fitting NB Classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)


## Fitting DT Classifier to the Training set
#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(criterion= 'entropy', random_state=0)
#classifier.fit(X_train,y_train)

## Fitting RF Classifier to the Training set
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
#classifier.fit(X_train,y_train)


## Fitting Logistic Regression to the Training set
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state = 0)
#classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#Checking accuracy
import sklearn.metrics as metrics
display(metrics.accuracy_score(y_test, y_pred))
display(metrics.precision_score(y_test, y_pred))
display(metrics.recall_score(y_test, y_pred))
display(metrics.f1_score(y_test, y_pred))




