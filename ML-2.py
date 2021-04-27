from pandas import read_csv
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

### features correlations
print("features correlations")
correlations = data.corr(method='pearson')
print(correlations)



