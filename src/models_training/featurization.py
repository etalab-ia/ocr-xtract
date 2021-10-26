import sys
import os
import yaml

import pandas as pd
from sklearn.pipeline import FeatureUnion

from src.augmentation.augmentation import AugmentDocuments
from src.preprocessing.xtract_vectorizer import WindowTransformerList, BoxPositionGetter, BagOfWordInLine
from src.preprocessing.word_transformers import ContainsDigit, IsPrenom, IsNom, IsDate


if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["featurize"]

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython featurization.py data-dir-path features-dir-path\n")
        sys.exit(1)

    train_input = os.path.join(sys.argv[1], "train.csv")
    test_input = os.path.join(sys.argv[1], "test.csv")
    data_output = os.path.join(sys.argv[2], "data")

    os.makedirs(os.path.join(sys.argv[2]), exist_ok=True)

    data_train = pd.read_csv(train_input, sep='\t')
    data_test = pd.read_csv(test_input, sep='\t')

    data_augmentation = params['data_augmentation']
    if data_augmentation:
        aug = AugmentDocuments()
        data_train = aug.transform(data_train)

    columns = data_train.columns.to_list()
    columns.remove('label')

    X_train, y_train = data_train[columns], data_train["label"]
    X_test, y_test = data_test[columns], data_test["label"]

    search_words = ['salaire', 'net', 'impots', 'periode', 'revenu', 'avant', 'sarl', 'sas', 'rue', 'monsieur',
                    'madame', 'm.', 'mme.', 'du', 'au']

    pipe_feature = FeatureUnion([('window_transformer', WindowTransformerList(searched_words=search_words)),
                                ('bag_of_words', BagOfWordInLine(searched_words=search_words)),
                                ('is_date', IsDate()),
                                ("position", BoxPositionGetter()),
                                ('is_digit', ContainsDigit()),
                                ('is_nom', IsNom()),
                                ('is_prenom', IsPrenom()),
                                ])

    X_train = pipe_feature.fit_transform(X_train)
    X_test = pipe_feature.transform(X_test)

    data = {
        "pipe_feature": pipe_feature,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }

    from pickle import dump
    with open(data_output, 'wb') as f1:
        dump(data, f1)




