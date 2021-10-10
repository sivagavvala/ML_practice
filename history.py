                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [1, 10, 100, 1000], 'kernel' : ['linear']},  # C - error paramter, increase in C prevent overfitting, if more high new problem - underfitting
              {'C' : [1, 10, 100, 1000], 'kernel' : ['rbf'], 'gamma' : [0.5, 0.1, 0.01, 0.001, 0.0001]}]

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [1, 10, 100, 1000], 'kernel' : ['linear']},  # C - error paramter, increase in C prevent overfitting, if more high new problem - underfitting
              {'C' : [1, 10, 100, 1000], 'kernel' : ['rbf'], 'gamma' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}]

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
from xgboost import XGBClassifier

## ---(Sun Nov 24 13:34:57 2019)---
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
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
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
cm.accuracy_score()
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
cm.accuracy_score(y_test,y_pred)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
X_train
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
accuracies.mean()
accuracies.std()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('train_LZdllcl.csv')
dataset = pd.read_csv('/home/siva/.config/spyder-py3/train_LZdllcl.csv')
head(dataset)
X = dataset.iloc[:, :13].values
y = dataset.iloc[:, 13].values

## ---(Mon Dec  9 08:56:40 2019)---
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('Data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 3].values
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues
missingvalues = missingvalues.fit(x[:, 1:3])
missingvalues
x[:, 1:3]=missingvalues.transform(x[:, 1:3])
x
missingvalues.statistics_
pd.Dataframe(x,x[:.1:3]).head()
pd.DataFrame(x,x[:.1:3]).head()
pd.DataFrame(x,x[:,1:3]).head()
x =pd.DataFrame(x,x[:,1:3]).head()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 3].values
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(x[:, 1:3])
x[:, 1:3]=missingvalues.transform(x[:, 1:3])
x
x =pd.DataFrame(x,x[:,1:3]).head()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 3].values
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(strategy = 'constant')
missingvalues = missingvalues.fit(x[:, 1:3])
x[:, 1:3]=missingvalues.transform(x[:, 1:3])
x
x =pd.DataFrame(x,x[:,1:3]).head()
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(strategy = 'constant',fill_value= '23948')
missingvalues = missingvalues.fit(x[:, 1:3])
x[:, 1:3]=missingvalues.transform(x[:, 1:3])
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(strategy = 'constant',fill_value= 23948)
missingvalues = missingvalues.fit(x[:, 1:3])
x[:, 1:3]=missingvalues.transform(x[:, 1:3])
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, 3].values
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(strategy = 'constant',fill_value= 23948)
missingvalues = missingvalues.fit(x[:, 1:3])
x[:, 1:3]=missingvalues.transform(x[:, 1:3])
x