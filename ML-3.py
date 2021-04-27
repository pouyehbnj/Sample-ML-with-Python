import pandas as pd
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
path = r".\imports-85.data"
headernames = ['symboling' , 'normalized-losses', 'make','fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels',
               'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders',
               'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
               'city-mpg', 'highway-mpg', 'price']
data = read_csv(path, names=headernames)


number = preprocessing.LabelEncoder()
data['make'] = number.fit_transform(data['make'])
data['fuel-type'] = number.fit_transform(data['fuel-type'])
data['aspiration'] = number.fit_transform(data.aspiration)
data['num-of-doors'] = number.fit_transform(data['num-of-doors'])
data['body-style'] = number.fit_transform(data['body-style'])
data['drive-wheels'] = number.fit_transform(data['drive-wheels'])
data['engine-location'] = number.fit_transform(data['engine-location'])
data['engine-type'] = number.fit_transform(data['engine-type'])
data['num-of-cylinders'] = number.fit_transform(data['num-of-cylinders'])
data['fuel-system'] = number.fit_transform(data['fuel-system'])
data=data.fillna(-999)

#DecisionTreeClassifier
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

#prediction
y_pred = classifier.predict(X_test)
print(y_pred)

### confusion_matrix
result = confusion_matrix(y_test, y_pred)
print(result)
print(classification_report(y_test, y_pred))