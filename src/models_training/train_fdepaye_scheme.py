from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

import pandas as pd
import os
from pathlib import Path
from pickle import load

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, SelectorMixin
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, Normalizer, PowerTransformer
from sklearn.decomposition import PCA

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


def select_candidate(candidate_name, candidate_feature, features, X, y=None):
    df = pd.DataFrame(X, columns=features)
    if y is not None:
        df['label'] = y.values
        df.loc[df['label'] != candidate_name, 'label'] = 0
        df.loc[df['label'] == candidate_name, 'label'] = 1
        # TODO : downsampled negative (max ratio 40 to 1)
    if candidate_feature is not None:
        df = df[df[candidate_feature] >= 1]
    if y is not None:
        X = df.drop(columns=['label']).to_numpy()
        y = df['label'].astype(int).to_numpy()
        return X, y
    else:
        X = df.to_numpy()
        return X


class tqdm_skopt(object):
    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)

    def __call__(self, res):
        self._bar.update()


def f(x):
    return (np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) *
            np.random.randn() * 0.1)

def train_field_model(candidate, data, n_cv=3, n_points=1, optimize=False):
    candidate_name = candidate['training_field']
    candidate_feature = candidate['candidate']
    pipe_feature = data['pipe_feature']
    X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']

    features = pipe_feature.get_feature_names()
    X_train, y_train = select_candidate(candidate_name, candidate_feature, features, X_train, y_train)
    X_test, y_test = select_candidate(candidate_name, candidate_feature, features, X_test, y_test)

    pipe_feature_post = Pipeline([
        ('power_transformer', PowerTransformer()),
    ])

    X_train = pipe_feature_post.fit_transform(X_train)
    X_test = pipe_feature_post.transform(X_test)

    n_iter = 100

    if optimize:
        pipe = BayesSearchCV(
            GradientBoostingClassifier(n_estimators=100, verbose=0),
            {
                'max_leaf_nodes': (10, 60),
                'learning_rate': (0.05, 0.2, 'uniform'),
                'max_depth': (3, 100),  # integer valued parameter
            },
            n_points=n_points,
            n_jobs=n_cv*n_points,
            n_iter=n_iter,
            cv=n_cv,
            verbose=0,
            scoring='f1_macro',
        )
        pipe.fit(X_train, y_train, callback=[tqdm_skopt(total=n_iter, desc=candidate_name)])

        print("val. score: %s" % pipe.best_score_)
    else:
        pipe = GradientBoostingClassifier(n_estimators=100, max_depth=80, max_leaf_nodes=40, learning_rate=0.1,
                                          verbose=1)
        pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    print(classification_report(y_test, y_pred))
    f1 = f1_score(y_test, y_pred)

    return pipe_feature_post, pipe, f1


if __name__ == "__main__":
    with open('./model/fdp_data_preprocessing_big_augment', 'rb') as f1:
        data = load(f1)

    scheme = {
        "somme": {
            'training_field': 'somme',
            'candidate': 'is_digit__ContainsDigit'
        },
        "date": {
            'training_field': 'date',
            'candidate': 'is_date__IsDate',
        },
        "nom": {
            'training_field': 'nom',
            'candidate': 'is_nom__IsNom'
        },
        "prenom": {
            'training_field': 'prenom',
            'candidate': 'is_prenom__IsPrenom'
        },
        "entreprise": {
            'training_field': 'entreprise',
            'candidate': None
        },
    }

    n_cpu = os.cpu_count()
    n_scheme = len(scheme.keys())
    n_cv = 3
    n_points = max(n_cpu // n_cv, 1)

    optimize = True

    if optimize:
        for candidate, candidate_name in zip(scheme.values(), scheme.keys()):
            scheme[candidate_name]['pipe_feature_post'], \
            scheme[candidate_name]['model'], \
            scheme[candidate_name]['f1'] = train_field_model(candidate,
                                                             data,
                                                             n_cv=n_cv,
                                                             n_points=n_points,
                                                             optimize=True)
    else:
        with Pool(n_scheme) as pool:
            res = list(
                    tqdm(pool.imap(partial(train_field_model, data=data), scheme.values()), total=n_scheme))

    from pickle import dump

    with open('./model/model_scheme_fdp_2', 'wb') as f1:
        dump(scheme, f1)
