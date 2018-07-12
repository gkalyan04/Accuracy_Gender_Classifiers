from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# height, weight and shoe size

X = [[181, 80, 10], [177, 70, 9], [160, 60, 8], [154, 54, 7], [166, 65, 6],
     [190, 90, 13], [175, 64, 9],
     [177, 70, 10], [159, 55, 7], [171, 75, 8], [181, 85, 9]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

tree_clf = DecisionTreeClassifier()
forest_clf = RandomForestClassifier()
log_clf = LogisticRegression()
KNN_clf = KNeighborsClassifier()

tree_clf = tree_clf.fit(X,Y)
forest_clf = forest_clf.fit(X,Y)
log_clf = log_clf.fit(X,Y)
KNN_clf = KNN_clf.fit(X,Y)

pred_tree_clf = tree_clf.predict(X)
acc_tree_clf = accuracy_score(pred_tree_clf,Y) * 100
print("Accuracy of DecisionTreeClassifier : %s" % (acc_tree_clf))

pred_forest_clf = forest_clf.predict(X)
acc_forest_clf = accuracy_score(pred_forest_clf, Y)* 100
print("Accuracy of RandomForestClassifier : %s" % (acc_forest_clf))

pred_log_clf = log_clf.predict(X)
acc_log_clf = accuracy_score(pred_log_clf, Y) * 100
print("Accuracy of LogisticRegression : %s" % (acc_log_clf))

pred_KNN_clf = KNN_clf.predict(X)
acc_KNN_clf = accuracy_score(pred_KNN_clf, Y) * 100
print("Accuracy of KNeighborsClassifier : %s" % (acc_KNN_clf))

# finding the classifier having maximum accuracy
index = np.argmax([acc_tree_clf, acc_forest_clf, acc_log_clf, acc_KNN_clf])
classfiers = {0: "DecisionTreeClassifier", 1: "RandomForestClassifier", 2: "LogisticRegression", 3: "KNeighborsClassifier"}
print(f"The highest accuracy is of {classfiers[index]}")



