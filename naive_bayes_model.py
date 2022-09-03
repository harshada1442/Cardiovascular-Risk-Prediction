import pandas as pd
from sklearn.metrics import accuracy_score, auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('data_cardiovascular_risk.csv')
#
# Drop all the rows with N.A as fields
data.dropna(inplace=True)
print(f"Size of data : {data.shape}")

dependent_variable = 'TenYearCHD'

all_independant_variables = ['age', 'education', 'sex',
                             'is_smoking', 'cigsPerDay', 'BPMeds',
                             'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI',
                             'heartRate',
                             'glucose']

independent_variables = ['age',
                         'cigsPerDay', 'BPMeds',
                         'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate',
                         'glucose']

# converting data suitable for model features
data['is_smoking'] = data['is_smoking'].replace({'YES': 1, 'NO': 0})
data['sex'] = data['sex'].replace({'F': 1, 'M': 0})

X = data[independent_variables]

y = data[dependent_variable]
# By using train_test_split we have split the data into traing dataset and testing datasets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)

# training the model on training set
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

# making predictions on the testing set
y_pred = gnb.predict(X_test)

# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics

print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)

