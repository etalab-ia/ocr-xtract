import pandas as pd
from pathlib import Path

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, Normalizer, PowerTransformer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline, FeatureUnion
from skopt import BayesSearchCV

from src.augmentation.augmentation import AugmentDocuments
from src.preprocessing.xtract_vectorizer import WindowTransformerList, BoxPositionGetter, BagOfWordInLine
from src.preprocessing.word_transformers import ContainsDigit, IsPrenom, IsNom, IsDate
import numpy as np


if __name__ == "__main__":
    data_train = pd.read_csv("./data/CORD/train/train.csv", sep='\t')
    data_test = pd.read_csv("./data/CORD/test/test.csv", sep='\t')

    # aug = AugmentDocuments()
    # data_train = aug.transform(data_train)

    data_train.loc[data_train['label'] != 'total.total_price', 'label'] = 'O'
    data_test.loc[data_test['label'] != 'total.total_price', 'label'] = 'O'

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
                                # ('is_nom', IsNom()),
                                # ('is_prenom', IsPrenom()),
                                ])

    X_train = pipe_feature.fit_transform(X_train)
    X_test = pipe_feature.transform(X_test)

    pipe_feature_post = Pipeline([
        ('power_transformer', PowerTransformer()),
    ])

    X_train = pipe_feature_post.fit_transform(X_train)
    X_test = pipe_feature_post.transform(X_test)

    # pipe = GradientBoostingClassifier(n_estimators=100, verbose=2)
    # pipe.fit(X_train, y_train)

    pipe = BayesSearchCV(
        GradientBoostingClassifier(n_estimators=100, verbose=2),
        {
            'max_leaf_nodes': (10, 60),
            'learning_rate': (0.05, 0.2, 'uniform'),
            'max_depth': (1, 100),  # integer valued parameter
        },
        n_jobs=-1,
        n_iter=32,
        cv=3,
        verbose=1,
        scoring='f1_macro'
    )
    pipe.fit(X_train, y_train)

    print("val. score: %s" % pipe.best_score_)

    y_pred = pipe.predict(X_test)

    print(classification_report(y_test, y_pred))

    from pickle import dump

    with open('./model/model_test_pipeline_cord', 'wb') as f1:
        dump(pipe_feature, f1)
        dump(pipe, f1)





