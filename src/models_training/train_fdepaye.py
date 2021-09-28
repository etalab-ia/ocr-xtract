from pickle import dump

import pandas as pd

from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion, Pipeline
import autosklearn.classification
from autosklearn.metrics import f1_macro

from src.preprocessing.xtract_vectorizer import WindowTransformerList, BoxPositionGetter, BagOfWordInLine
from src.preprocessing.word_transformers import ContainsDigit, IsPrenom, IsNom, IsDate

if __name__ == "__main__":
    data_train = pd.read_csv("./data/salary_for_training/train_annotated.csv", sep='\t')
    data_test = pd.read_csv("./data/salary_for_training/test_annotated.csv", sep='\t')

    columns = data_train.columns.to_list()
    columns.remove('label')

    X_train, y_train = data_train[columns], data_train["label"]
    X_test, y_test = data_test[columns], data_test["label"]

    search_words = ['salaire','net','impots','periode','revenu','avant','sarl','sas','rue']

    pipe_feature = FeatureUnion([('window_transformer', WindowTransformerList(searched_words=search_words)),
                                ('bag_of_words', BagOfWordInLine(searched_words=search_words)),
                                ('is_date', IsDate()),
                                ("position", BoxPositionGetter()),
                                ('is_digit', ContainsDigit()),
                                ('is_nom', IsNom()),
                                ('is_prenom', IsPrenom()),
                                ])

    X_train = pipe_feature.fit_transform(X_train)
    df = pd.DataFrame(X_train, columns=pipe_feature.get_feature_names())
    df['word'] = data_train["word"] #for debug
    X_test = pipe_feature.transform(X_test)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=43200,
        per_run_time_limit=180,
        memory_limit=None,
        n_jobs=16,
        metric=f1_macro,
    )
    automl.fit(X_train, y_train)

    print(automl.sprint_statistics())

    predictions = automl.predict(X_test)

    print(classification_report(y_test, predictions))

    with open('./model/fdp_model_automl', 'wb') as f2:
        dump(pipe_feature,f2)
        dump(automl, f2)
