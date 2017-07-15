from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import accuracy_score

def basicnn():
    from sklearn.neural_network import MLPClassifier
    X = [[0., 0.], [1., 1.]]
    y = [0, 1]
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X, y)
    a  = clf.predict([[2., 2.], [-1., -2.]])
    a  = clf.predict([[0., 1.], [-1., -2.]])
    print(a)

def cancernn():
    
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    print(cancer.keys())
    X = cancer['data']
    y = cancer['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(X_train)
    # Now apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
    mlp.fit(X_train,y_train)
    predictions = mlp.predict(X_test)
    from sklearn.metrics import classification_report,confusion_matrix
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    len(mlp.coefs_)
    len(mlp.coefs_[0])
    len(mlp.intercepts_[0])

def harnn():
    har = pd.read_table('dataset-har.txt',sep= ';')
    har['how_tall_in_meters'] = har['how_tall_in_meters'].apply(lambda x: x.replace(',', '.') )
    har['body_mass_index'] = har['body_mass_index'].apply(lambda x: x.replace(',', '.') )
    #har2 = har.select_dtypes(include = [numpy.number])
    X = har[['x1','y1','z1','x2','y2','z2','x3','y3','z3','x4','y4','z4']]
    y = har['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=111)
    mlp = MLPClassifier(hidden_layer_sizes=(5,10,5), random_state = 777, verbose = True)
    mlp.fit(X_train,y_train)
    predictions = mlp.predict(X_test)
    cm = accuracy_score(y_test, predictions)
    print(cm)
    from sklearn.metrics import classification_report,confusion_matrix
    #print(confusion_matrix(y_test,predictions))
    #print(classification_report(y_test,predictions))
    len(mlp.coefs_)
    len(mlp.coefs_[0])
    len(mlp.intercepts_[0])


    
