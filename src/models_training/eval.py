import os
import sys
from pickle import load
import json

from sklearn.metrics import f1_score

from src.models_training.utils import select_candidate


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython eval.py features-dir-path scheme-path model-dir-path results-dir-path\n")
        sys.exit(1)

    features_folder = os.path.join(sys.argv[1], "data.pickle")
    pipe_file = os.path.join(sys.argv[1], "pipe.pickle")
    scheme_name = sys.argv[2].split('/')[1]
    scheme_path = sys.argv[2]
    model_folder = sys.argv[3]
    results_folder = sys.argv[4]
    os.makedirs(results_folder, exist_ok=True)

    # load data and scheme
    with open(features_folder, 'rb') as f1:
        data = load(f1)
    with open(scheme_path, 'rb') as f_s:
        scheme = json.load(f_s)
    with open(pipe_file, 'rb') as f1:
        pipe_feature = load(f1)

    X_test, y_test = data['X_test'], data['y_test']
    features = pipe_feature.get_feature_names()

    list_model = os.listdir(model_folder)

    results = {}

    for candidate_name in scheme.keys():
        print(candidate_name)
        candidate_feature = scheme[candidate_name]['candidate']

        list_model_candidate = [m for m in list_model if candidate_name == m.split('-')[0]]

        results[candidate_name] = {}

        for model_name in list_model_candidate:
            with open(os.path.join(model_folder, model_name), 'rb') as f:
                model_data = load(f)

            # pipe_feature_post = model_data['pipe_feature_post']
            model = model_data['model']

            X, y, _ = select_candidate(candidate_name, candidate_feature, features, X_test, y_test)
            # X = pipe_feature_post.transform(X)
            y_pred = model.predict(X)

            f1 = f1_score(y, y_pred)
            print(f1)

            results[candidate_name][model_name] = f1

    with open(os.path.join(results_folder, 'results.json'), 'w') as f:
        json.dump(results, f)