from sklearn.cluster import KMeans
from pandas import read_csv
from sklearn import preprocessing
import matplotlib.pyplot as plt


path = r".\imports-85.data"
headernames = ['symboling' , 'normalized-losses', 'make','fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels',
               'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders',
               'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm',
               'city-mpg', 'highway-mpg', 'price']
dataset = read_csv(path, names=headernames)
number = preprocessing.LabelEncoder()
dataset['make'] = number.fit_transform(dataset['make'])
##dataset['fuel-type'] = number.fit_transform(dataset['fuel-type'])
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
X = dataset.drop('fuel-type', axis=1)
Y = dataset['fuel-type']

kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c = y_kmeans, s = 20, cmap = 'summer')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = 'blue', s = 100, alpha = 0.9);
plt.show()
