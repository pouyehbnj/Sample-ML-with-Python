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

path = r".\imports-85.data"
headernames = ['symboling' , 'normalized-losses', 'make','fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels',
               'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders',
               'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
               'city-mpg', 'highway-mpg', 'price']
dataset = read_csv(path, names=headernames)
number = preprocessing.LabelEncoder()
dataset['make'] = number.fit_transform(dataset['make'])
dataset['fuel-type'] = number.fit_transform(dataset['fuel-type'])
dataset['aspiration'] = number.fit_transform(dataset.aspiration)
dataset['num-of-doors'] = number.fit_transform(dataset['num-of-doors'])
dataset['body-style'] = number.fit_transform(dataset['body-style'])
dataset['drive-wheels'] = number.fit_transform(dataset['drive-wheels'])
dataset['engine-location'] = number.fit_transform(dataset['engine-location'])
dataset['engine-type'] = number.fit_transform(dataset['engine-type'])
dataset['num-of-cylinders'] = number.fit_transform(dataset['num-of-cylinders'])
dataset['fuel-system'] = number.fit_transform(dataset['fuel-system'])



# dataset['engine-size'] = number.fit_transform(dataset.engine-size)
dataset=dataset.fillna(-999)
X = dataset.drop('price', axis=1)
Y = dataset['price']


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# ### confusion_matrix
# result = confusion_matrix(y_test, y_pred)
# print(result)
regressor = LinearRegression()

regressor.fit(X_train, y_train)
print(regressor.intercept_)

y_pred = regressor.predict(X_test)
print(y_pred)

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, Y)
# dataset.plot(x='make', y='price', style='o')
# plt.title('Hours vs Percentage')
# plt.xlabel('Hours Studied')
# plt.ylabel('Percentage Score')
# plt.show()

plt.scatter(X_poly, Y,color='g')
plt.plot(X, pol_reg.predict(X),color='k')

plt.show()