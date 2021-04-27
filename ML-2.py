from pandas import read_csv
import seaborn as sns
from matplotlib import pyplot


path = r".\imports-85.data"
headernames = ['symboling','normalized-losses','make','num-of-doors','body-style','drive-wheels',
'engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders',
'engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm',
'city-mpg','highway-mpg','price']
data = read_csv(path, names=headernames)
print(data.head(50))

#### shape of data
print("shape of data")
print(data.shape)

#### class distribution
print("class distribution")
for headername in headernames:
    count_class = data.groupby(headername).size()
    print(count_class)

#### features types
print("features types")
print(data.dtypes)

### features correlations
print("features correlations")
correlations = data.corr(method='pearson')
print(correlations)

### feature scatter plot

print("feature scatter plot")



### density diagram
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.show()







