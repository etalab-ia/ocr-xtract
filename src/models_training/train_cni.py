import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

from src.salaire.doctr_utils import WindowTransformerList
import numpy as np


data_train = pd.read_csv("./data/CNI_recto/cni_annotation_recto_train.csv", sep='\t')
data_test = pd.read_csv("./data/CNI_recto/cni_annotation_recto_test.csv", sep=';')
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

classifier = OneVsRestClassifier(LinearSVC(random_state=0))
classifier.fit(np.transpose(X_train), y_train)
y_pred = classifier.predict(np.transpose(X_test))

print(f"f1: {f1_score(y_test, y_pred, average='weighted')}")
print(f"precision_score: {precision_score(y_test, y_pred, average='weighted')}")
print(f"recall_score: {recall_score(y_test, y_pred, average='weighted')}")



classifier = OneVsRestClassifier(DecisionTreeClassifier(random_state=0))
classifier.fit(np.transpose(X_train), y_train)
y_pred = classifier.predict(np.transpose(X_test))

print(f"f1: {f1_score(y_test, y_pred, average='weighted')}")
print(f"precision_score: {precision_score(y_test, y_pred, average='weighted')}")
print(f"recall_score: {recall_score(y_test, y_pred, average='weighted')}")


classifier = OneVsRestClassifier(RandomForestClassifier(random_state=0))
classifier.fit(np.transpose(X_train), y_train)
y_pred = classifier.predict(np.transpose(X_test))

print(f"f1: {f1_score(y_test, y_pred, average='weighted')}")
print(f"precision_score: {precision_score(y_test, y_pred, average='weighted')}")
print(f"recall_score: {recall_score(y_test, y_pred, average='weighted')}")
