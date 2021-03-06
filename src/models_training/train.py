import sys
import os
import yaml
import json
from pickle import load, dump
from tqdm import tqdm

from skopt import BayesSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from src.models_training.utils import select_candidate, tqdm_skopt


def prepare_data(candidate, data, features):
    candidate_name = candidate['training_field']
    candidate_feature = candidate['candidate']
    X_train, y_train = data['X_train'], data['y_train']
    X_train, y_train, _ = select_candidate(candidate_name, candidate_feature, features, X_train, y_train)

    return X_train, y_train



if __name__ == "__main__":

    if len(sys.argv) != 5:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython train.py features-dir-path scheme-path model-dir-path param_name\n")
        sys.exit(1)

    train_input = os.path.join(sys.argv[1], "data.pickle")
    pipe_file = os.path.join(sys.argv[1], "pipe.pickle")
    scheme_file = sys.argv[2]

    os.makedirs(os.path.join(sys.argv[3]), exist_ok=True)



    # load data scheme and features
    with open(train_input, 'rb') as f1:
        data = load(f1)
    with open(scheme_file, 'rb') as f_s:
        scheme = json.load(f_s)
    with open(pipe_file, 'rb') as f1:
        pipe_feature = load(f1)
    features = pipe_feature.get_feature_names()

    params = yaml.safe_load(open("params.yaml"))[sys.argv[4]]

    optimize = params['optimize']
    n_iter = params['n_iter']
    n_estimators = params['n_estimators']

    if not optimize:
        learning_rate = params['learning_rate']
        max_depth = params['max_depth']
        max_leaf_nodes = params['max_leaf_nodes']

    n_scheme = len(scheme.keys())
    n_cpu = os.cpu_count()
    n_cv = 3  # cross validation for optimization
    n_points = max(n_cpu // n_cv, 1)

    for candidate, candidate_name in tqdm(zip(scheme.values(), scheme.keys())):

        X_train, y_train = prepare_data(candidate, data, features)
        # pipe_feature_post = Pipeline([
        #     ('power_transformer', PowerTransformer()),
        # ])
        #
        # X_train = pipe_feature_post.fit_transform(X_train)
        model = {}
        # model['pipe_feature_post'] = pipe_feature_post

        if optimize:
            model_output = os.path.join(sys.argv[3], candidate_name + '-' + "optimized_GB")
            pipe = BayesSearchCV(
                GradientBoostingClassifier(n_estimators=n_estimators, verbose=0),
                {
                    'max_leaf_nodes': (10, 60),
                    'learning_rate': (0.05, 0.2, 'uniform'),
                    'max_depth': (3, 100),  # integer valued parameter
                },
                n_points=n_points,
                n_jobs=n_cv * n_points,
                n_iter=n_iter,
                cv=n_cv,
                verbose=0,
                scoring='f1_macro',
            )
            pipe.fit(X_train, y_train)

            print("val. score: %s" % pipe.best_score_)
            model['model'] = pipe

        else:
            model_output = os.path.join(sys.argv[3], candidate_name + '-' + "GB")
            gbc = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators,
                                max_depth=max_depth, verbosity=1, n_jobs=n_cpu,
                                eval_metric=f1_score)
            gbc.fit(X_train, y_train)
            model['model'] = gbc

        with open(model_output, 'wb') as f1:
            dump(model, f1)


