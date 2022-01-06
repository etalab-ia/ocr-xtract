import sys
import os
import yaml
import json
from pickle import load, dump

from skopt import BayesSearchCV


if __name__ == "__main__":

    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython train.py scheme-path model-dir-path output_path\n")
        sys.exit(1)

    scheme_file = sys.argv[1]
    model_dir_path = sys.argv[2]
    list_model = os.listdir(model_dir_path)

    # load scheme
    with open(scheme_file, 'rb') as f_s:
        scheme = json.load(f_s)

    params = yaml.safe_load(open("params.yaml"))["train"]


    for candidate_name in scheme.keys():
        candidate_feature = scheme[candidate_name]['candidate']
        list_model_candidate = [m for m in list_model if candidate_name == m.split('-')[0] and "optimized_GB" in m]
        for model_name in list_model_candidate:
            with open(os.path.join(model_dir_path, model_name), 'rb') as f_m:
                model = load(f_m)
                bayes = model['model']
                print(bayes.best_params_)
