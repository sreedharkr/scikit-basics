import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

har = pd.read_table('dataset-har.txt',sep= ';')
print(har.columns.tolist())

import matplotlib.pyplot as plt
#hardf['x1'].plot(kind = 'density')
#plt.show()

har['how_tall_in_meters'] = har['how_tall_in_meters'].apply(lambda x: x.replace(',', '.') )
har['body_mass_index'] = har['body_mass_index'].apply(lambda x: x.replace(',', '.') )
X = har.iloc[:,:18]
y = har.iloc[:,18]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=111)

X1 = X_train.iloc[:,6:18]
#y = .iloc[:,18]
y1 = y_train
print("sample target values:::", y1[1:5])

#print(har.head()
start = time.clock()
clf = RandomForestClassifier(n_estimators=100, max_depth=None,
                             min_samples_split=5, random_state=111)
clf.fit(X1,y1)
predicted = clf.predict(X_test.iloc[:, 6:18])
#cm = confusion_matrix(y, predicted)
cm = accuracy_score(y_test, predicted)
print(cm)
print(time.clock() - start)
#scores = cross_val_score(clf, X, y)
#print( scores.mean() )                       




    
