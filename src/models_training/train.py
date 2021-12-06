import sys
import os
import yaml
import json
from pickle import load, dump

from skopt import BayesSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline

from src.models_training.utils import select_candidate, tqdm_skopt


def train_optimized_model(candidate, data, n_cv=3, n_points=1, n_iter=100, n_estimators=100):
    candidate_name = candidate['training_field']
    candidate_feature = candidate['candidate']
    pipe_feature = data['pipe_feature']
    X_train, y_train = data['X_train'], data['y_train']

    features = pipe_feature.get_feature_names()
    X_train, y_train = select_candidate(candidate_name, candidate_feature, features, X_train, y_train)

    pipe_feature_post = Pipeline([
        ('power_transformer', PowerTransformer()),
    ]) # TODO : do we really need to do that here ?

    X_train = pipe_feature_post.fit_transform(X_train)

    pipe = BayesSearchCV(
        GradientBoostingClassifier(n_estimators=n_estimators, verbose=0),
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
    pipe.fit(X_train, y_train, callback=[tqdm_skopt(total=int(n_iter/(n_cv*n_points)), desc=candidate_name)])

    print("val. score: %s" % pipe.best_score_)

    return pipe_feature_post, pipe


if __name__ == "__main__":

    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython train.py features-dir-path scheme-path model-dir-path\n")
        sys.exit(1)

    train_input = os.path.join(sys.argv[1], "data.pickle")
    scheme_file = sys.argv[2]

    os.makedirs(os.path.join(sys.argv[3]), exist_ok=True)

    # load data and scheme
    with open(train_input, 'rb') as f1:
        data = load(f1)
    with open(scheme_file, 'rb') as f_s:
        scheme = json.load(f_s)

    params = yaml.safe_load(open("params.yaml"))["train"]

    n_iter = params['n_iter']
    n_estimators = params['n_estimators']


    n_scheme = len(scheme.keys())
    n_cpu = os.cpu_count()
    n_cv = 3  # cross validation for optimization
    n_points = max(n_cpu // n_cv, 1)
    for candidate, candidate_name in zip(scheme.values(), scheme.keys()):
        model_output = os.path.join(sys.argv[3], candidate_name + '-' + "optimized_GB")
        model = {}
        model['pipe_feature_post'], \
        model['model'] = train_optimized_model(candidate,
                                               data,
                                               n_cv=n_cv,
                                               n_points=n_points,
                                               n_iter=n_iter,
                                               n_estimators=n_estimators)

        with open(model_output, 'wb') as f1:
            dump(model, f1)


