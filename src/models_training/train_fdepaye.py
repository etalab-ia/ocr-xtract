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

from tpot import TPOTClassifier


if __name__ == "__main__":
    data_train = pd.read_csv("./data/salary_for_training/train_annotated.csv", sep='\t')
    data_test = pd.read_csv("./data/salary_for_training/test_annotated.csv", sep='\t')

    data_train['word'] = data_train['word'].apply(lambda x: str(x).encode('utf-8'))
    data_test['word'] = data_test['word'].apply(lambda x: str(x).encode('utf-8'))

    columns = data_train.columns.to_list()
    columns.remove('label')

    X_train, y_train = data_train[columns], data_train["label"]
    X_test, y_test = data_test[columns], data_test["label"]

    pipe_feature = FeatureUnion([('window_transformer', WindowTransformerList(searched_words=['salaire','net','impots','periode','revenu'])),
                                ('bag_of_words', BagOfWordInLine(searched_words=['salaire','net','impots','periode','revenu'])),
                                ('is_date', IsDate()),
                                ("position", BoxPositionGetter()),
                                ('is_digit', ContainsDigit()),
                                ('is_nom', IsNom()),
                                ('is_prenom', IsPrenom()),
                                ])



    X_train = pipe_feature.fit_transform(X_train)
    df = pd.DataFrame(X_train, columns=pipe_feature.get_feature_names())
    X_test = pipe_feature.transform(X_test)

    tpot = TPOTClassifier(generations=5, population_size=20,  verbosity=2, random_state=42, n_jobs=-1)
    tpot.fit(X_train, y_train)

    y_pred = tpot.predict(X_test)
    X_test['label']=y_pred

    print(classification_report(y_test, y_pred))

    tpot.export('./model/tpot_exported_pipeline.py')

    """
    y_pred = pipe.predict(X_test)
    X_test['label']=y_pred
    print(X_test)
    print(classification_report(y_test, y_pred))
    
    from pickle import dump
    with open('./model/model_test_h', 'wb') as f1:
        dump(pipe, f1)
    """


