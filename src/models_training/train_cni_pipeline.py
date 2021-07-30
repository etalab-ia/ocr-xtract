import pandas as pd
from pathlib import Path

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline, FeatureUnion

from src.salaire.doctr_utils import WindowTransformerList, BoxPositionGetter, ContainsDigit
import numpy as np


data_train = pd.read_csv("./data/CNI_recto_aligned_linux/annotation_train.csv", sep='\t')
data_test = pd.read_csv("./data/CNI_recto_aligned_linux/annotation_test.csv", sep='\t')
columns = data_train.columns.to_list()
columns.remove('label')
X_train, y_train = data_train[columns], data_train["label"]
X_test, y_test = data_test[columns], data_test["label"]


pipe = Pipeline([
    ('feature_union', FeatureUnion([('window_transformer', WindowTransformerList(searched_words=['Nom:','Prénom(s):','Né(e)'])),
                                    ("position", BoxPositionGetter()),
                                    ('is_digit', ContainsDigit())
                                    ])),
    ('decision_tree', GradientBoostingClassifier())
])

# lb = LabelBinarizer()
# y_train = lb.fit_transform(y_train)
# y_test = lb.transform(y_test)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))

from pickle import dump
with open('./model/CNI_model', 'wb') as f1:
    dump(pipe, f1)
# with open('./model/CNI_label', 'wb') as f2:
#     dump(lb, f2)


