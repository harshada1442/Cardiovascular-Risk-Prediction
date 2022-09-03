import pandas as pd
from sklearn.metrics import accuracy_score, auc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# Fit on Train Set

tenYearCHD_classifier = DecisionTreeClassifier(max_depth=4,max_leaf_nodes=7, random_state=0)
tenYearCHD_classifier.fit(X_train, y_train)

### Measure Accuracy of the Classifier
# Predict on Train Set
y_train_predicted = tenYearCHD_classifier.predict(X_train)
print(f"Training Accuracy : {accuracy_score(y_train, y_train_predicted) * 100}")

# Predict on Test Set
y_predicted = tenYearCHD_classifier.predict(X_test)
print(f"Testing Accuracy : {accuracy_score(y_test, y_predicted) * 100}")
