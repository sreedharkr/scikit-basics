from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
iris = load_iris()
dir(iris)
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 111)
arry1 = np.arange(2,10)
cv_scores = []
for a in arry1:
    knn = KNeighborsClassifier(n_neighbors = a)
    scores = cross_val_score(knn, X_train, y_train, cv = 10, scoring = "accuracy")
    cv_scores.append(scores.mean())
print(cv_scores)
print("max accuracy is ::", np.max(cv_scores))
max_index = cv_scores.index(max(cv_scores))
print("maximum accuracy for k = ", arry1[max_index])

def cross1():
    from sklearn import datasets, linear_model
    from sklearn.model_selection import cross_val_score
    diabetes = datasets.load_diabetes()
    X = diabetes.data[:150]
    y = diabetes.target[:150]
    lasso1 = linear_model.Lasso()
    print(cross_val_score(lasso1, X, y))  # doctest: +ELLIPSIS

    X1 = diabetes.data
    y1 = diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.33)
    lasso1.fit(X_train, y_train)
    print("lasso fitted object  ", lasso1)
    
