from tqdm import tqdm

import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd

from jenkspy import JenksNaturalBreaks


def goodness_of_variance_fit(array, classes):
    # get the break points
    jen = JenksNaturalBreaks(classes)
    jen.fit(array)

    # sum of squared deviations from array mean
    sdam = np.sum((array - array.mean()) ** 2)

    # do the actual classification
    groups = jen.group(array)

    # sum of squared deviations of class means
    sdcm = sum([np.sum((group - group.mean()) ** 2) for group in groups])

    # goodness of variance fit
    gvf = (sdam - sdcm) / sdam

    return gvf

def get_optimal_nb_classes(y):
    gvf = 0.0
    nclasses = 3
    while gvf < .9999:
        if nclasses > int(len(y) / 2):
            break
        gvf = goodness_of_variance_fit(y, nclasses)
        if gvf < .9999:
            nclasses += 1
        if gvf == 1.0:
            if nclasses == 3:
                return nclasses
            nclasses -= 1
    return nclasses



class BoxPositionGetter(TransformerMixin, BaseEstimator):
    """
    Transforms the box position given with min_x, max_y, min_y, max_y in two values x y being the center of the box
    """
    def fit(self, X, y=None):
        return self

    def get_feature_names(self):
        return ["middle_x","middle_y"]

    def find_middle(self, X):
        middle_x = (X['min_x'] + X["max_x"])/2
        middle_y = (X['min_y'] + X["max_y"])/2
        middle_x = middle_x.to_numpy()
        middle_y = middle_y.to_numpy()
        return np.vstack([middle_x,middle_y]).T

    def transform(self, X):
        print(f"Transforming with {self.__class__.__name__}")
        return self.find_middle(X)




class LinesClustering(TransformerMixin, BaseEstimator):
    """
    This vectorizer clusters the document by lines with Jenks Natural Breaks.
    It then finds all the words contained in a line a calculate a vector with CountVectorizer
    The vector gets calculated for every word in the document
    """


    def _transform(self, doct_documents: pd.DataFrame):
        print(f"Transforming {self.__class__.__name__}")

        self.array_lines = np.zeros(doct_documents.shape[0])
        i = 0
        for doc in tqdm(doct_documents['document_name'].unique()):
            for page_id, page in enumerate(doct_documents[doct_documents['document_name'] == doc]['page_id'].unique()):
                df = doct_documents[(doct_documents['document_name'] == doc) & (doct_documents['page_id'] == page)].copy()

                y = df['max_y'] * 100

                if len(y) > 3: #TODO check if >2 works here. Normally it should be the case
                    nb_class = get_optimal_nb_classes(y)

                    jnb = JenksNaturalBreaks(nb_class=nb_class)

                    if len(y) > nb_class:
                        jnb.fit(y)
                        predicted_lines = jnb.predict(y)
                        df['line'] = predicted_lines
                    else:
                        predicted_lines = [0 for i in y]
                        df['line'] = predicted_lines
                else:
                    predicted_lines = [0 for i in y]
                    df['line'] = predicted_lines

                for line in predicted_lines:
                    self.array_lines[i] = line

        self.array = self.array_lines.T
        self.feature_names_ = ['jenks_lines']

        return np.transpose(self.array)

    def transform(self, X: pd.DataFrame):
        return self._transform(X)
