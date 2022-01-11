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
    :return: the features and the labelled data with only the candidates selected,
            the list of indexes where the candidates are located
    """
    df = pd.DataFrame(X, columns=features)
    if y is not None:
        df['label'] = y.values
        df.loc[df['label'] != candidate_name, 'label'] = 0
        df.loc[df['label'] == candidate_name, 'label'] = 1
        # TODO : downsampled negative (max ratio 40 to 1)
    if candidate_feature != 'None':
        is_candidate = df[candidate_feature] >= 1
        df = df[is_candidate]
        is_candidate = is_candidate[is_candidate].index.values
    else:
        is_candidate = df.index.values
    if y is not None:
        X = df.drop(columns=['label']).to_numpy()
        y = df['label'].astype(int).to_numpy()
        return X, y, is_candidate
    else:
        X = df.to_numpy()
        return X, is_candidate


class tqdm_skopt(object):
    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)

    def __call__(self, res):
        self._bar.update()

    def __getstate__(self):
        # don't save away the temporary pbar_ object which gets created on
        # epoch begin anew anyway. This avoids pickling errors with tqdm.
        state = self.__dict__.copy()
        del state['_bar']
        return state