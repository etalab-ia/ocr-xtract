from tqdm import tqdm
import pandas as pd

def select_candidate(candidate_name, candidate_feature, features, X, y=None):
    """
    This function is used to select the words that can be candidates for a certain field.

    :param candidate_name: the name of the candidate. This must be the name used during annotation
    :param candidate_feature: this is the name of the feature used for the candidate selection
    :param features: the name of the features
    :param X: the features data
    :param y: the labelled data (optionnal)
    :return: the features and the labelled data with only the candidates selected
    """
    df = pd.DataFrame(X, columns=features)
    if y is not None:
        df['label'] = y.values
        df.loc[df['label'] != candidate_name, 'label'] = 0
        df.loc[df['label'] == candidate_name, 'label'] = 1
        # TODO : downsampled negative (max ratio 40 to 1)
    if candidate_feature != 'None':
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