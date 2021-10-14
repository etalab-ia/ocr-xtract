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

from src.augmentation.augmentation import AugmentDocuments
from src.preprocessing.xtract_vectorizer import WindowTransformerList, BoxPositionGetter, BagOfWordInLine
from src.preprocessing.word_transformers import ContainsDigit, IsPrenom, IsNom, IsDate
import numpy as np


if __name__ == "__main__":
    data_train = pd.read_csv("./data/salary_for_training/train_annotated.csv", sep='\t')
    data_test = pd.read_csv("./data/salary_for_training/test_annotated.csv", sep='\t')

    aug = AugmentDocuments()
    data_train = aug.transform(data_train)

    columns = data_train.columns.to_list()
    columns.remove('label')

    X_train, y_train = data_train[columns], data_train["label"]
    X_test, y_test = data_test[columns], data_test["label"]

    search_words = ['salaire','net','impots','periode','revenu','avant','sarl','sas','rue','monsieur','madame','m.','mme.','du','au']

    pipe_feature = FeatureUnion([('window_transformer', WindowTransformerList(searched_words=search_words)),
                                ('bag_of_words', BagOfWordInLine(searched_words=search_words)),
                                ('is_date', IsDate()),
                                ("position", BoxPositionGetter()),
                                ('is_digit', ContainsDigit()),
                                ('is_nom', IsNom()),
                                ('is_prenom', IsPrenom()),
                                ])

    X_train = pipe_feature.fit_transform(X_train)
    X_test = pipe_feature.transform(X_test)

    data = {
        "pipe_feature": pipe_feature,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }

    from pickle import dump
    with open('./model/fdp_data_preprocessing', 'wb') as f1:
        dump(data, f1)




