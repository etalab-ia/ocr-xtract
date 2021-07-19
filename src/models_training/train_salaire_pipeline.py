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


pipe = Pipeline([
    ('window_transformer', WindowTransformerList()),
    ('decision_tree', DecisionTreeClassifier())
])

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))

from pickle import dump
with open('./model/salaire_model', 'wb') as f1:
    dump(pipe, f1)
with open('./model/salaire_label', 'wb') as f2:
    dump(lb, f2)

