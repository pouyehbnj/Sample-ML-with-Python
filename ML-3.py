import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
path = r".\imports-85.data"
headernames = ['symboling', 'normalized-losses', 'make', 'num-of-doors', 'body-style', 'drive-wheels',
               'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders',
               'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
               'city-mpg', 'highway-mpg', 'price']
data = read_csv(path, names=headernames)


number = preprocessing.LabelEncoder()
data['make'] = number.fit_transform(data['make'])
data['num-of-doors'] = number.fit_transform(data['num-of-doors'])
data['body-style'] = number.fit_transform(data['body-style'])
data['drive-wheels'] = number.fit_transform(data['drive-wheels'])
data['engine-location'] = number.fit_transform(data['engine-location'])
data['engine-type'] = number.fit_transform(data['engine-type'])
data['num-of-cylinders'] = number.fit_transform(data['num-of-cylinders'])
data['fuel-system'] = number.fit_transform(data['fuel-system'])
data=data.fillna(-999)


X = data.drop('price', axis=1)
Y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)

# myData=np.genfromtxt(path, delimiter=",", dtype ="|a20" ,skip_header=1)
# le = preprocessing.LabelEncoder()
# X = myData.data[:, :4]
# y = myData.target
# for i in range(26):
#     myData[:,i] = le.fit_transform(myData[:,i])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
print(X_train)