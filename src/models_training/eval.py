import os
import sys
from pickle import load
import json

from sklearn.metrics import f1_score
from tqdm import tqdm

import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

from src.models_training.utils import select_candidate

load_dotenv()

def prepare_mlflow_server():
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_SERVER_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            tqdm.write(f"MLflow tracking to server {tracking_uri}")
            pass
        else:
            tqdm.write(f"MLflow tracking to local mlruns folder")
    except Exception as e:
        tqdm.write(f"Not using remote tracking servers. Error {e}")
        tqdm.write(f"MLflow tracking to local mlruns folder")



if __name__ == "__main__":
    prepare_mlflow_server()
    client = MlflowClient()

    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython eval.py features-dir-path scheme-path model-dir-path\n")
        sys.exit(1)

    features_folder = os.path.join(sys.argv[1], "data.pickle")
    scheme_name = sys.argv[2].split('/')[1]
    scheme_path = sys.argv[2]
    model_folder = os.path.join(sys.argv[3])

    # load data and scheme
    with open(features_folder, 'rb') as f1:
        data = load(f1)
    with open(scheme_path, 'rb') as f_s:
        scheme = json.load(f_s)

    pipe_feature = data['pipe_feature']
    X_test, y_test = data['X_test'], data['y_test']
    features = pipe_feature.get_feature_names()

    list_model = os.listdir(model_folder)

    for candidate_name in scheme.keys():
        print(candidate_name)
        candidate_feature = scheme[candidate_name]['candidate']

        list_model_candidate = [m for m in list_model if candidate_name == m.split('-')[0]]

        expe = scheme_name + '_' + candidate_name
        mlflow.set_experiment(experiment_name=expe)
        experiment = client.get_experiment_by_name(expe)

        for model_name in list_model_candidate:
            with open(os.path.join(model_folder, model_name), 'rb') as f:
                model_data = load(f)

            pipe_feature_post = model_data['pipe_feature_post']
            model = model_data['model']

            X, y = select_candidate(candidate_name, candidate_feature, features, X_test, y_test)
            X = pipe_feature_post.transform(X)
            y_pred = model.predict(X)

            f1 = f1_score(y, y_pred)
            print(f1)

            run = client.create_run(experiment.experiment_id)
            client.log_param(run.info.run_id, 'model', model_name)
            client.log_metric(run.info.run_id, 'f1', f1)
            client.log_artifact(run.info.run_id, 'dvc.lock')
