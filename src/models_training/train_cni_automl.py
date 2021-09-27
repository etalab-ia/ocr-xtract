import autosklearn.classification
from autosklearn.metrics import f1_macro
import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

from sklearn.pipeline import Pipeline, FeatureUnion

from src.salaire.doctr_utils import WindowTransformerList, BoxPositionGetter, ContainsDigit, IsNom, IsPrenom, IsDate
from pickle import dump

if __name__ == "__main__":
    data_train = pd.read_csv("./data/cni_recto_csv_for_training/train_annotated.csv", sep='\t')
    data_test = pd.read_csv("./data/cni_recto_csv_for_training/test_annotated.csv", sep='\t')

    columns = data_train.columns.to_list()
    columns.remove('label')

    X_train, y_train = data_train[columns], data_train["label"]
    X_test, y_test = data_test[columns], data_test["label"]


    pipe_feature = FeatureUnion([('window_transformer', WindowTransformerList(searched_words=['salaire','net','impots','periode','revenu'])),
                                # ('bag_of_words', BagOfWordInLine(searched_words=['salaire','net','impots','periode','revenu'])),
                                ('is_date', IsDate()),
                                ("position", BoxPositionGetter()),
                                ('is_digit', ContainsDigit()),
                                ('is_nom', IsNom()),
                                ('is_prenom', IsPrenom()),
                                ])


    X_train = pipe_feature.fit_transform(X_train)
    df = pd.DataFrame(X_train, columns=pipe_feature.get_feature_names())
    X_test = pipe_feature.transform(X_test)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=60,
        per_run_time_limit=60,
        memory_limit=None,
        n_jobs=10,
        include={
            'classifier': [
                'decision_tree', 'lda', 'gaussian_nb', 'random_forest'
            ],
        },
        metric=f1_macro

        # # Bellow two flags are provided to speed up calculations
        # # Not recommended for a real implementation
        # initial_configurations_via_metalearning=0,
        # smac_scenario_args={'runcount_limit': 1},

    )
    automl.fit(X_train, y_train)


    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(automl.sprint_statistics())

    predictions = automl.predict(X_test)

    print(classification_report(y_test, predictions))




    with open('./model/CNI_model_automl', 'wb') as f1:
        dump(automl, f1)

