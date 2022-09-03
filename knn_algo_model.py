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

#feature Scaling
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(X_train)
x_test= st_x.transform(X_test)

#Fitting K-NN classifier to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )
classifier.fit(x_train, y_train)

#Predicting the test set result
y_pred= classifier.predict(x_test)

#Creating the Confusion matrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)
print(cm)
