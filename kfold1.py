from sklearn import datasets
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
 
bcdata1 = load_breast_cancer()
features = bcdata1.data
target = bcdata1.target
arr1 = np.c_[features,target]
bcdf = pd.DataFrame(data = arr1)
kf_total = KFold(n_splits=5, shuffle=True, random_state=4)
print(type(kf_total))
print(dir(kf_total))
splits = kf_total.split(bcdf)
type(splits) #generator

lasso = Lasso(alpha = 0.1)

listk = [a for a in splits]
for a in listk:
    X_train = bcdf.loc[a[0],:]
    print(X_train.shape)
    y_train = bcdf.loc[a[1],:]
    print(y_train.shape)
    print(X_train.iloc[:,0:30].shape)
    train1 = X_train.iloc[:,0:30]
    test1 = X_train.iloc[:,30]
    #print(type(X_train))
    print(cross_val_score(lasso,train1,test1))

