from sklearn import datasets
import pandas as pd
boston_data = datasets.load_boston()
print('testing')
dir(boston_data)
bdf = np.c_[boston_data.data, boston_data.target]
bdf_features = np.append(boston_data.feature_names,np.array(['target']))
bdf1 = pd.DataFrame(data= bdf , columns = bdf_features)
# selecting rows using filter method

#how mant types of indexing are there
