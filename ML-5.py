from sklearn.cluster import KMeans
from pandas import read_csv
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt





path = r".\imports-85.data"
headernames = ['symboling' , 'normalized-losses', 'make','fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels',
               'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders',
               'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
               'city-mpg', 'highway-mpg', 'price']
dataset = read_csv(path, names=headernames)
# number = preprocessing.LabelEncoder()
# dataset['make'] = number.fit_transform(dataset['make'])
# dataset['fuel-type'] = number.fit_transform(dataset['fuel-type'])
# dataset['aspiration'] = number.fit_transform(dataset.aspiration)
# dataset['num-of-doors'] = number.fit_transform(dataset['num-of-doors'])
# dataset['body-style'] = number.fit_transform(dataset['body-style'])
# dataset['drive-wheels'] = number.fit_transform(dataset['drive-wheels'])
# dataset['engine-location'] = number.fit_transform(dataset['engine-location'])
# dataset['engine-type'] = number.fit_transform(dataset['engine-type'])
# dataset['num-of-cylinders'] = number.fit_transform(dataset['num-of-cylinders'])
# dataset['fuel-system'] = number.fit_transform(dataset['fuel-system'])



# dataset['engine-size'] = number.fit_transform(dataset.engine-size)
# dataset=dataset.fillna(-999)
X = np.array(dataset['highway-mpg']).reshape(-1, 1)
Y = np.array(dataset['curb-weight']).reshape(-1, 1)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

regressor = LinearRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test,color='g')
plt.plot(X_test, y_pred,color='k')

plt.show()

rms = sqrt(mean_squared_error(y_pred,y_test))


print(f'rms in linear is {rms}')

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, Y)

y_pred = pol_reg.predict(X_poly)
# plt.figure(figsize=(10,8))
# plt.scatter(X, Y)
# plt.plot(X, y_pred)
# print(r2_score(Y, y_pred))
# plt.show()
plt.scatter(X, Y,color='g')
plt.plot(X, y_pred,color='k')

plt.show()

rms = mean_squared_error(Y, y_pred, squared=False)

print(f'rms in polynomial is {rms}')