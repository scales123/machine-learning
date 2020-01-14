# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_boston
from sklearn import tree
from sklearn.model_selection import train_test_split

# 分类树
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
print ("Classifier Score:", clf.score(X_test, y_test))

tree.plot_tree(clf.fit(X, y))
plt.show()

# 回归树
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)
print ("Regression Score:", clf.score(X_test, y_test))

tree.plot_tree(clf.fit(X, y))
plt.show()
