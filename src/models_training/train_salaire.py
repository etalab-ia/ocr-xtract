import pandas as pd
from pathlib import Path

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline

from src.salaire.doctr_utils import WindowTransformerList
import numpy as np


data_train = pd.read_csv("./data/salary/salaires_annotation_train.csv", sep=';')
data_test = pd.read_csv("./data/salary/salaires_annotation_test.csv", sep=';')
columns = data_train.columns.to_list()
columns.remove('label')
X_train, y_train = data_train[columns], data_train["label"]
X_test, y_test = data_test[columns], data_test["label"]


lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

wt = WindowTransformerList()
X_train = wt.fit_transform(X_train)
X_test = wt.transform(X_test)

X_train_angle = X_train[:,:int(X_train.shape[1] / 2)] #take first half that corresponds to the angles
X_test_angle = X_test[:,:int(X_train.shape[1] / 2)]

X_train_distance = X_train[:,int(X_train.shape[1] / 2):] #take first half that corresponds to the angles
X_test_distance = X_test[:,int(X_train.shape[1] / 2):]

X_train_is_left = np.where(np.logical_or(np.pi -0.1 < X_train_angle, X_train_angle < -np.pi + 0.1), 1.0, 0.0)
X_test_is_left = np.where(np.logical_or(np.pi -0.1 < X_test_angle, X_test_angle < -np.pi + 0.1), 1.0, 0.0)


# X_train = np.concatenate([X_train_distance, X_train_is_left], axis=1)
# X_test = np.concatenate([X_test_distance, X_test_is_left], axis=1)



# classifier = OneVsRestClassifier(LinearSVC(random_state=0))
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
#
# print(f"f1: {f1_score(y_test, y_pred, average='macro')}")
# print(f"precision_score: {precision_score(y_test, y_pred, average='macro')}")
# print(f"recall_score: {recall_score(y_test, y_pred, average='macro')}")


# classifier = OneVsRestClassifier(DecisionTreeClassifier(random_state=0))
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
#
# print(classification_report(y_test, y_pred))
# print(f"f1: {f1_score(y_test, y_pred, average='macro')}")
# print(f"precision_score: {precision_score(y_test, y_pred, average='macro')}")
# print(f"recall_score: {recall_score(y_test, y_pred, average='macro')}")
#
classifier = (DecisionTreeClassifier(random_state=0))
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
#
# print(f"f1: {f1_score(y_test, y_pred, average='macro')}")
# print(f"precision_score: {precision_score(y_test, y_pred, average='macro')}")
# print(f"recall_score: {recall_score(y_test, y_pred, average='macro')}")
#
#
# classifier = OneVsRestClassifier(RandomForestClassifier(random_state=0))
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
#
# print(f"f1: {f1_score(y_test, y_pred, average='macro')}")
# print(f"precision_score: {precision_score(y_test, y_pred, average='macro')}")
# print(f"recall_score: {recall_score(y_test, y_pred, average='macro')}")
#
# y_pred = lb.inverse_transform(y_pred)
# np.savetxt("./data/CNI_recto/y_pred.csv", y_pred, delimiter=",", fmt='%s')


