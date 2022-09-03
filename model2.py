# importing libraries
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd

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

# Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10)

# feature Scaling
from sklearn.preprocessing import StandardScaler

st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)

# 2. Fitting the Random Forest algorithm to the training set:
# Now we will fit the Random forest algorithm to the training set. To fit it, we will import the RandomForestClassifier class from the sklearn.ensemble library. The code is given below:


# Fitting Decision Tree classifier to the training set
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=8, )
classifier.fit(x_train, y_train)

# Predicting the test set result
y_pred = classifier.predict(x_test)

# Creating the Confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
