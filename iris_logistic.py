import sklearn as sk
from sklearn import datasets,metrics,model_selection, cluster
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from ggplot import *

def logistic1():
    iris = sk.datasets.load_iris()
    iris_train = iris.data
    print(iris_train.shape)
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
         X, y, test_size=0.33, random_state=42)
    # Lasso l1(reduces to zero) Ridge L2(square) (default in sklearn is L2)
    # Inverse of regularization strength; must be a positive float.
    # Like in support vector machines, smaller values specify stronger
    #  regularization.
    print(X_train.shape,y_train.shape)
    logistic1 = LogisticRegression(penalty = "l1", C = 0.5)
    #logistic1 = LogisticRegression(multi_class = 'multinomial',solver ='newton-cg')
    #logistic1 = LogisticRegression(multi_class = 'multinomial', penalty = 'l2',
    #                               solver ='newton-cg')
    logistic1.fit(X_train,y_train)
    pred_values = logistic1.predict(X_test)
    cmatrix = confusion_matrix(y_test, pred_values)
    print(cmatrix)
    print(logistic1.coef_)
    print("penalty function",logistic1.penalty, sep = "  ")
    from sklearn.cross_validation import cross_val_score
    scores = cross_val_score(logistic1, X, y, scoring='accuracy', cv=10)
    print(scores)
#corss_val_score LogisticRegressionCV GridSearchCV
def logisticcv1():
    from sklearn.cross_validation import cross_val_score
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn import datasets,metrics,model_selection
    import numpy as np
    from collections import Counter
    iris = sk.datasets.load_iris()
    iris_train = iris.data
    print(iris_train.shape)
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
         X, y, test_size=0.33, random_state=42)
    print(X_train.shape,y_train.shape)
    #alphas = np.logspace(-2, -0.5, 30)
    alphas = np.arange(0.4,0.55,0.001)
    #logisticcv = LogisticRegressionCV(cv = 10, penalty = 'l1', solver = 'liblinear')
    logistic2 = LogisticRegression(penalty = 'l1', solver = 'liblinear')
    scores = list()
    maxscore = 0
    maxa = 0
    for a in alphas:
        #print(a)
        logistic2.C = a
        scores2 = cross_val_score(logistic2,X_train, y_train, cv = 10)
        mean1 = scores2.mean()
        print("alpha and score", a, mean1)
        if mean1 > maxscore:
            maxscore = mean1
            maxa = a
        #tup1 = tuple( (a,scores2.mean()))
        #scores.append(tup1)
        #print(scores)
        #print("Below cross_val_score train dataset",scores2, sep = "\n")
        #print("scores2 mean  ", scores2.mean())
    #this is not cv
    #list2 = [ b for b in scores]
    print(maxa, maxscore)
    logistic2.C = 0.5
    c1 = Counter(y_test)
    print(c1)
    logistic2.fit(X_train,y_train)
    pred_values = logistic2.predict(X_test)
    cmatrix = confusion_matrix(y_test, pred_values)
    print("below is confusion matrix on train data set",cmatrix, sep = "\n")
    print("coefficieinets  ",logistic2.coef_)
    print("penalty function",logistic2.penalty, sep = "  ")
    #scores = cross_val_score(logistic2, X, y, scoring='accuracy', cv=10)
    #print(scores)
def logisticcv2():
    from sklearn.cross_validation import cross_val_score
    #from sklearn.model_selection import KFold
    from sklearn.cross_validation import KFold #this is outdated use above
    iris = sk.datasets.load_iris()
    iris_train = iris.data
    print(iris_train.shape)
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
         X, y, test_size=0.33, random_state=42)
    #fold = KFold(n_splits=5, shuffle=True, random_state=777)
    fold = KFold(len(y_train), n_folds=5, shuffle=True, random_state=777)
    #print(dir(fold))
    #print(X_train.shape,y_train.shape)
    #logistic2 = LogisticRegressionCV(cv = 10, penalty = 'l1', solver = 'liblinear')
    #scoring = "accuracy" roc_auc
    list2 = np.arange(0.4,0.75,0.001)
    logistic2 = LogisticRegressionCV(cv = 5,scoring = "accuracy",
                                     Cs=list2,
                                     penalty = 'l2',
                                     random_state = 313, max_iter = 100, tol = 10)
    logistic23 = LogisticRegressionCV(cv = 5,scoring = "accuracy",random_state = 313)
    #logistic2.C = 0.50
    logistic2.fit(X_train,y_train)
    print(logistic2)
    print(dir(logistic2))
    #print(logistic2.Cs_)
    #print(np.max(logistic2.scores_))
    pred_values = logistic2.predict(X_test)
    cmatrix = confusion_matrix(y_test, pred_values)
    print("below is confusion matrix on train data set",cmatrix, sep = "\n")

def cancer_log():
    cancer  = datasets.load_breast_cancer()
    print(dir(cancer))
    X = cancer.data
    y = cancer.target
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
         X, y, test_size=0.33, random_state=42)
    print("training dataset size",X_train.shape)
    

    
def pandas_iris():
    #from imp import reload
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    iris = load_iris()
    print(type(iris))
    irispd = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
    print(type(irispd))
    print(irispd.head())
    gp= ggplot(irispd, aes('sepal length (cm)', 'sepal width (cm)')) +geom_point()
    gp= ggplot(irispd, aes('sepal length (cm)', 'sepal width (cm)',colour = 'target')) + geom_point()
    #gp2 = ggplot(irispd, aes('target', 'sepal width (cm)',fill = 'target')) + geom_boxplot()
    gp2 = ggplot(irispd, aes('target', 'sepal width (cm)')) + geom_boxplot(fill = "blue", colour = "red")
    #print(gp)
    print(gp2)
    iris_sub = irispd.ix[:,0:4]
    from pandas.tools.plotting import scatter_matrix
    pd.scatter_matrix(iris_sub)
    plt.show()

def dbscan_iris():
    #from imp import reload
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    iris = load_iris()
    print(type(iris))
    irispd = pd.DataFrame(data= iris['data'],
                     columns = iris['feature_names'])
    print(type(irispd))
    print(irispd.head())
    irispd = irispd[["sepal length (cm)", "sepal width (cm)"]]
    irispd = irispd.as_matrix().astype("float32", copy = False)
    stscaler = StandardScaler().fit(irispd)
    irispd = stscaler.transform(irispd)
    dbsc = DBSCAN(eps = .5, min_samples = 15).fit(irispd)
    labels = dbsc.labels_
    print(labels)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    core_samples = np.zeros_like(labels, dtype = bool)
    #print("core_samples ::",core_samples)
    core_samples[dbsc.core_sample_indices_] = True
    #print("core_samples ::",core_samples)
    import matplotlib.pyplot as plt
    unique_labels = set(labels)
    print('unique_labels',unique_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    print('colors::',colors)
    for k, col in zip(unique_labels, colors):
        print("k  col  ", k ,col)
        if k == -1:
        # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)
        print('class_member_mask', class_member_mask)
        #xy = X[class_member_mask & core_samples_mask]
        xy = irispd[class_member_mask & core_samples]
        print(":::::::::::::::::::::::::::::::::::::::")
        print("xy.shape::" , xy.shape)
        a = xy[:,0:1]
        b = xy[:,1:2]
        print("shape of a  ",a.shape)
        print("shape of b  ",b.shape)
        plt.plot(a, b, 'o')
        #, markerfacecolor=col.any())
                 #markeredgecolor='k', markersize=14)
        print("???????????????????????????????????")
        xy = irispd[class_member_mask & ~core_samples]
        plt.plot(xy[:, 0], xy[:, 1], 'o')
        xy = irispd[class_member_mask & ~core_samples]
        plt.plot(xy[:, 0], xy[:, 1], 'o')
                 #, markerfacecolor=col.any(),
                 # markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def misc():
    df = pd.DataFrame(np.random.randn(100, 4), columns=['a', 'b', 'c', 'd'])
    axes = scatter_matrix(df, alpha=0.5, diagonal='kde')
    corr = df.corr().as_matrix()
    for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
        axes[i, j].annotate("%.3f" %corr[i,j], (0.8, 0.8), xycoords='axes fraction', ha='center', va='center')
    plt.show()
