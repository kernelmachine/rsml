from sklearn.ensemble import RandomForestClassifier
import numpy as np

def foo():
	X = np.random.rand(5,10)
	Y = np.random.choice([0,1],size = 5)
	clf = RandomForestClassifier(n_estimators = 30)
	clf = clf.fit(X,Y)


