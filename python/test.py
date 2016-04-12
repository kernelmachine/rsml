from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def foo():
	X = np.random.rand(500,10000)
        Y = np.random.choice([0,1],size = 500)
	rf = RandomForestClassifier(n_estimators = 1000)
	rf.fit(X,Y)
	import time
	start = time.clock()
	rf.predict(X)
	print time.clock() - start

def bar():
        X = np.random.rand(2,2)
        Y = np.random.choice([0,1],size = 2)
	dt = DecisionTreeClassifier()
	dt.fit(X,Y)
	dt.predict(X)

foo()
