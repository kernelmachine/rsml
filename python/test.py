from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def foo():
	X = np.random.rand(5,10)
	Y = np.random.choice([0,1],size = 5)
	clf = RandomForestClassifier(n_estimators = 30)
	clf = clf.fit(X,Y)

def bar():
	X = np.random.rand(50,50)
	Y = np.random.choice([0,1], size = 50)
	clf = DecisionTreeClassifier()
	clf = clf.fit(X,Y)

