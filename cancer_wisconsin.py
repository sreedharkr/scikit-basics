# diff scale vs normalize(divide by vector length, l1,l2) vs minmax_scale

import sklearn
from sklearn import preprocessing
import pandas as pd
cancer_data = sklearn.datasets.load_breast_cancer()
fnames = cancer_data.feature_names
df = pd.DataFrame(data = cancer_data.data, columns = fnames)
df.dtypes

dict2 = df1.columns.to_series().groupby(df1.dtypes).groups
# only float types
df11 = df.loc[:, df.dtypes == np.float64]
df1.loc[:,('mean radius','mean texture')]
dff = df1.iloc[:,0:30]

df1.columns[0], df1.columns[1]
for b in df1.columns:
    print(b)

print (df.head())
df3 = ( df - df.mean()) / df.std()
# using sklearn
#df2 = preprocessing.StandardScaler().fit_transform(df)
preprocessing.scale(df3)

#list(df3.columns.values) or list(df3)
#loc works on lables, ilod on index , loc as iloc
df1.loc[1:5, ('mean area','texture error')]
df1.iloc[1:5, [1,2]]
df1.columns # returns indexer immutable ndarray
#ix is deprecated from 0.20 and above
print(df3.ix[:,:5].head())
print(df3.describe())
print(df3.ix[:,:8].corr())
print(df3.ix[:,8:16].corr())
print(df3.ix[:,16:25].corr())
print(df3.ix[:,25:30].corr())

#nan values
from numpy import nan
df1.iloc[100,:].to_frame().T
df11 = df1.dropna(how = 'any')
df1.iloc[100,3] = nan



