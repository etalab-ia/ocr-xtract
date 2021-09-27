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

from src.salaire.doctr_utils import WindowTransformerList, BoxPositionGetter, ContainsDigit, IsNom, IsPrenom, IsDate, \
    BagOfWordInLine
import numpy as np


if __name__ == "__main__":
    data_train = pd.read_csv("./data/salary_for_training/train_annotated.csv", sep='\t')
    data_test = pd.read_csv("./data/salary_for_training/test_annotated.csv", sep='\t')

    columns = data_train.columns.to_list()
    columns.remove('label')

    X_train, y_train = data_train[columns], data_train["label"]
    X_test, y_test = data_test[columns], data_test["label"]

    pipe = Pipeline([
        ('feature_union', FeatureUnion([('window_transformer', WindowTransformerList(searched_words=['salaire','net','impots','periode','revenu'])),
                                        ('bag_of_words', BagOfWordInLine(searched_words=['salaire','net','impots','periode','revenu'])),
                                        # ('is_date', IsDate()),
                                        ("position", BoxPositionGetter()),
                                        ('is_digit', ContainsDigit()),
                                        ('is_nom', IsNom()),
                                        ('is_prenom', IsPrenom()),
                                        ])),
        ('decision_tree', GradientBoostingClassifier())
    ])


    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(classification_report(y_test, y_pred))

    from pickle import dump
    with open('./model/model_test_h', 'wb') as f1:
        dump(pipe, f1)



