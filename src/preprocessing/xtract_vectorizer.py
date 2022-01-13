import difflib
from functools import partial
from typing import List
import re

from tqdm import tqdm

import multiprocessing as mp

import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from stop_words import get_stop_words

from jenkspy import JenksNaturalBreaks


# TODO : add [NUMBER], [RARE] and [PAD] in vectorizer (PAD???)

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
    nclasses = 2
    while gvf < .9999:
        gvf = goodness_of_variance_fit(y, nclasses)
        if gvf < .9999:
            nclasses += 1
        if gvf == 1.0:
            nclasses -= 1
        if nclasses > int(len(y) / 2):
            break
    return nclasses


class XtractVectorizer(DictVectorizer):
    """ This call is used for vectorizing text extracted from DocTr

    Parameters
    ----------
    searched_words : List[str], default=None
        Specify here if you are looking for some specific words in the document that could be considered as anchors
    n_jobs : int, default=-1
        Indicates the number of threads you want to use when a parallelized process is available
    min_df : float in range [0.0, 1.0] or int, default=0.2
        Used to be passed to CountVectorizer
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    Attributes
    ----------


    """
    def __init__(self, searched_words: List[str] = None, n_jobs: int = -1, min_df: float = 0.2):
        super().__init__(sparse=False)
        self.searched_words = searched_words
        not_stop_words = set(['ne','le','nom','nommé','nommée','nommés','du','au'])
        not_stop_words.update(searched_words)
        additional_stop_word = ['nan','ca', 'debut', 'etaient', 'etais', 'etait', 'etant', 'etat', 'ete', 'etes', 'etiez', 'etions', 'etre', 'eumes', 'eutes', 'fumes', 'futes', 'meme', 'tres']
        self.stop_words = get_stop_words('french').copy()
        self.stop_words.extend(additional_stop_word)
        for word in not_stop_words:
            try:
                self.stop_words.remove(word)
            except:
                pass
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else min(n_jobs, mp.cpu_count())
        self.min_df = min_df
        self._list_words_in_page = []

    def get_words_in_page(self, df, for_vectorizer=False):
        if self.searched_words is None:
            list_words = [str(doc) for doc in df['word'].to_list()]
        else:
            list_words = []
            for doc in df['word'].to_list():
                doc = str(doc)
                close_matches = difflib.get_close_matches(doc, self.searched_words)
                if len(close_matches) > 0:
                    list_words.append(close_matches[0])
                else:
                    list_words.append(doc)
        if for_vectorizer: #remove the strings that contain digits
            list_ind = []
            for i, word in enumerate(list_words):
                if len(re.findall(r"\d", str(word))) > 0:
                    list_ind.append(i)
            for i in list_ind[::-1]:
                list_words.pop(i)
        return list_words

    def fit(self, doctr_documents: pd.DataFrame, **kwargs):
        min_df = int(self.min_df * len(doctr_documents['document_name'].unique()))
        self.list_words = self.get_words_in_page(doctr_documents, for_vectorizer=True)
        self.vectorizer = CountVectorizer(strip_accents='ascii', min_df=min_df, stop_words=self.stop_words)
        self.vectorizer.fit(self.list_words)
        self.vocab = self.vectorizer.get_feature_names()
        return self


    def fit_transform(self, X: pd.DataFrame,y = None, **kwargs):
        self.fit(X)
        return self._transform(X)

class WindowTransformerList(XtractVectorizer):

    def get_middle_position(self, df, i):
        return (df.iloc[i]["min_x"] + df.iloc[i]["max_x"]) / 2, (df.iloc[i]["min_y"] + df.iloc[i]["max_y"]) / 2

    def get_relative_positions(self, vocab_i, list_plain_words_in_page, df):
        # TODO : the calculation of the distances between the words of the vocab is done twice, we should avoid that
        array_angles = np.ones(len(list_plain_words_in_page))  # false value for angle
        array_distances = np.ones(len(list_plain_words_in_page)) # max distance
        if vocab_i in list_plain_words_in_page:
            wi_list = [i for i, x in enumerate(list_plain_words_in_page) if x == vocab_i]
            for wi in wi_list:
                for j, word_j in enumerate(list_plain_words_in_page):
                    x_i, y_i = df.iloc[wi]['min_x'], df.iloc[wi]['min_y']
                    x_j, y_j = df.iloc[j]['min_x'], df.iloc[j]['min_y']
                    distance = cosine((x_i, y_i), (x_j, y_j))
                    if distance < array_distances[j]:  # in case there are several identical duplicate of vocab i, take the closest
                        array_angles[j] = np.arctan2((y_j - y_i), (x_j - x_i)) / np.pi
                        array_distances[j] = distance
        return array_angles, array_distances

    def treat_doc(self, doc, X):
        for page_id, page in enumerate(X[X['document_name'] == doc]['page_id'].unique()):
            df = X[(X['document_name'] == doc) & (X['page_id'] == page)]
            list_plain_words_in_page = self.get_words_in_page(df)
            list_token = self.vectorizer.inverse_transform(self.vectorizer.transform(list_plain_words_in_page))
            for i, token in enumerate(list_token):
                if len(token) > 0:
                    list_plain_words_in_page[i] = token[0]
            self._list_words_in_page.extend(list_plain_words_in_page)
            vocab = self.vocab
            res = []
            for voc in vocab:
                res.append(
                    self.get_relative_positions(vocab_i=voc, list_plain_words_in_page=list_plain_words_in_page, df=df))
            array_angle = np.array([r[0] for r in res]).T
            array_distance = np.array([r[1] for r in res]).T
            return array_angle, array_distance


    def _transform(self, X: pd.DataFrame, **kwargs):
        print(f"Transforming {self.__class__.__name__}")
        print(f"vocab that will be used for transform {self.vocab}")
        if len(X['document_name'].unique()) > 1: #this is for training or testing
            nb_cpu = self.n_jobs
        else: #this is for the inference
            nb_cpu = 1 #limit the number of cpu to 1 to avoid memory issues in production
        print(f'Number of CPU used by {self.__class__.__name__}: {nb_cpu}')
        with mp.Pool(nb_cpu) as pool:
            arrays = list(tqdm(pool.imap(partial(self.treat_doc, X=X), X['document_name'].unique()), total=len(X['document_name'].unique())))


        list_array_angle = [a[0] for a in arrays]
        list_array_distance = [a[1] for a in arrays]

        array_angle = np.vstack(list_array_angle)
        array_distances = np.vstack(list_array_distance)
        array = np.concatenate([array_angle.T, array_distances.T])
        self.feature_names_ = [str(a) + '_angle' for a in self.vocab] + [str(a) + '_distance' for a in self.vocab]

        # TODO : keep the name of the doc and pages in a self.list_doc self.list_pages
        return array.T


    def transform(self, X: pd.DataFrame):
        return self._transform(X)



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


class BagOfWordInLine(XtractVectorizer):
    """
    This vectorizer clusters the document by lines with Jenks Natural Breaks.
    It then finds all the words contained in a line a calculate a vector with CountVectorizer
    The vector gets calculated for every word in the document
    """


    def _transform(self, doct_documents: pd.DataFrame, **kwargs):
        print(f"Transforming {self.__class__.__name__}")
        print(f"vocab that will be used for transform {self.vocab}")

        self.array_lines = np.zeros(doct_documents.shape[0])
        self.array_bows = np.zeros((doct_documents.shape[0], len(self.vectorizer.get_feature_names())))
        i = 0
        for doc in tqdm(doct_documents['document_name'].unique()):
            for page_id, page in enumerate(doct_documents[doct_documents['document_name'] == doc]['page_id'].unique()):
                df = doct_documents[(doct_documents['document_name'] == doc) & (doct_documents['page_id'] == page)].copy()

                y = df['max_y'] * 100

                nb_class = get_optimal_nb_classes(y)

                jnb = JenksNaturalBreaks(nb_class=nb_class)
                jnb.fit(y)

                predicted_lines = jnb.predict(y)
                df['line'] = predicted_lines

                for line in predicted_lines:
                    words = [str(w) for w in df[df['line'] == line]['word'].to_list()]
                    bag = ' '.join(words)
                    self.array_lines[i] = line
                    self.array_bows[i] = self.vectorizer.transform([bag]).toarray()
                    i += 1

        self.array = np.vstack((self.array_lines.T, self.array_bows.T))
        self.feature_names_ = ['lines'] +[f'bag_of_words_{f}' for f in self.vectorizer.get_feature_names()]

        # TODO : keep the name of the doc and pages in a self.list_doc self.list_pages
        return np.transpose(self.array)

    def transform(self, X: pd.DataFrame):
        return self._transform(X)





