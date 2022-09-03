
# import libraries
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from jedi.api.refactoring import inline


data = pd.read_csv('data_cardiovascular_risk.csv')

# Drop all the rows with N.A as fields
data.dropna(inplace=True)
print(f"Size of data : {data.shape}")

# converting data suitable for model features
data['is_smoking'] = data['is_smoking'].replace({'YES': 1, 'NO': 0})
data['sex'] = data['sex'].replace({'F': 1, 'M': 0})

dependent_variable = ['TenYearCHD']

all_independant_variables = ['age', 'education', 'sex',
                             'is_smoking', 'cigsPerDay', 'BPMeds',
                             'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI',
                             'heartRate',
                             'glucose']

# independent_variables = ['age',
#                          'cigsPerDay', 'BPMeds',
#                          'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate',
#                          'glucose']

independent_variables = ['age',
                         'cigsPerDay',
                         'prevalentHyp', 'diabetes', 'sysBP', 'diaBP', 'BMI', 'heartRate',
                         'glucose']

# Extracting Independent and dependent Variable
x = data[independent_variables]
y = data[dependent_variable]

from sklearn.model_selection import train_test_split

# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import train_test_split

# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score, confusion_matrix

target = y
features = x

X_train, X_test, y_train, y_test = train_test_split(features,target, test_size = 0.2, random_state = 8)
from sklearn.svm import SVC

# Building a Support Vector Machine on train data
svc_model = SVC(C=.1, kernel='linear', gamma=1)
svc_model.fit(X_train, y_train)

prediction = svc_model.predict(X_test)
# check the accuracy on the training set
print(svc_model.score(X_train, y_train))
print(svc_model.score(X_test, y_test))

print("Confusion Matrix:\n",confusion_matrix(prediction,y_test))

svc_model = SVC(kernel='sigmoid')
svc_model.fit(X_train, y_train)

prediction = svc_model.predict(X_test)

print(svc_model.score(X_train, y_train))
print(svc_model.score(X_test, y_test))