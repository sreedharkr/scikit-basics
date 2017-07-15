import pandas as pd
import time
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

print('hello')
har = pd.read_table('dataset-har.txt',sep= ';')
print(har.columns.tolist())
har.info()

hardf.dtypes
#change the column type
#har['x1'] = har['x1'].astype(float)
#har.obj = har.select_dtypes(include = [np.object])

har['how_tall_in_meters'] = har['how_tall_in_meters'].apply(lambda x: x.replace(',', '.') )
har['body_mass_index'] = har['body_mass_index'].apply(lambda x: x.replace(',', '.') )

#har.describe()
print (har.describe(include = [numpy.number]) )
print(har.x1.describe())

har.query('x1 > 10')

har_numeric = har.select_dtypes(include = [numpy.number])
har_numeric.shape
har_numeric12 = har_numeric.iloc[:, 2:14]
# what is patsy

cmatrix = hardf_numeric.corr(method = 'pearson')
print( type(cmatrix) )
cmatrix2 =  cmatrix > 0.5
np.count_nonzero(cmatrix2)
np.sum(cmatrix2)

s = cmatrix2.unstack()
so = s.sort_values()
so[so == True]


import matplotlib.pyplot as plt

#feature_selection

SelectKBest ch2 cannot be applied because of negative values in dataset

from sklearn.feature_selection import SelectKBest, f_classif
selector2 = SelectKBest(f_classif,k = 'all')
har_numeric12 = har_numeric.iloc[:, 2:14]
selector2.fit(har_numeric12, hardf['class'])
selector2.scores_
selector2.get_support()

#Mutual information (MI) [R169] between two random variables is a
#non-negative value, which measures the dependency between the variables.
#It is equal to zero if and only if two random variables are independent,
#and higher values mean higher dependency. 
from sklearn.feature_selection import mutual_info_classif
selector3 = SelectKBest(mutual_info_classif, k = 'all') #k = 9
selector3.fit(hardf_numeric12, hardf['class'])
selector3.scores_

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
iris = load_iris()
X, y = iris.data, iris.target
X.shape
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
logistic1 = LogisticRegression(penalty = 'l1').fit(X,y)
#model = SelectFromModel(lsvc, prefit=True)
model = SelectFromModel(logistic1, prefit=True)
X_new = model.transform(X)
X_new.shape




           
