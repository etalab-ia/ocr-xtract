import pandas as pd
from pathlib import Path

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, Normalizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline, FeatureUnion

from src.preprocessing.xtract_vectorizer import WindowTransformerList, BoxPositionGetter, BagOfWordInLine
from src.preprocessing.word_transformers import ContainsDigit, IsPrenom, IsNom, IsDate
import numpy as np


if __name__ == "__main__":
    data_train = pd.read_csv("./data/salary_for_training/train_annotated.csv", sep='\t')
    data_test = pd.read_csv("./data/salary_for_training/test_annotated.csv", sep='\t')

    columns = data_train.columns.to_list()
    columns.remove('label')

    X_train, y_train = data_train[columns], data_train["label"]
    X_test, y_test = data_test[columns], data_test["label"]

    search_words = ['salaire','net','impots','periode','revenu','avant','sarl','sas','rue']


    pipe = Pipeline([
        ('feature_union', FeatureUnion([('window_transformer', WindowTransformerList(searched_words=search_words)),
                                ('bag_of_words', BagOfWordInLine(searched_words=search_words)),
                                ('is_date', IsDate()),
                                ("position", BoxPositionGetter()),
                                ('is_digit', ContainsDigit()),
                                ('is_nom', IsNom()),
                                ('is_prenom', IsPrenom()),
                                ('norm', Normalizer(norm='l2', copy=False)),
                                ])),
        ('decision_tree', GradientBoostingClassifier(verbose=1))
    ])


    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(classification_report(y_test, y_pred))

    from pickle import dump
    with open('./model/model_test_h', 'wb') as f1:
        dump(pipe, f1)



