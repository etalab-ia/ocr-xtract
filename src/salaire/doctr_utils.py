import sys
import difflib
from datetime import datetime
from collections import Counter
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Optional, List, Dict
import re

from joblib import Parallel, delayed
from tqdm import tqdm

import swifter
from memory_profiler import profile

import multiprocessing as mp

import numpy as np
from doctr.models import ocr_predictor
from doctr.documents import DocumentFile, Page, Document, Word
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from stop_words import get_stop_words

from jenkspy import JenksNaturalBreaks


import dateparser


class PageExtended(Page):
    pass

class XtractVectorizer(DictVectorizer):
    """ This call is used for vectorizing text extracting from DocTr

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
        self.n_jobs = n_jobs
        self.min_df = min_df
        self._list_words_in_page = []

    def get_words_in_page(self, df):
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
        return list_words

    def fit(self, doctr_documents: pd.DataFrame, **kwargs):
        min_df = int(self.min_df * len(doctr_documents['document_name'].unique()))
        self.list_words = self.get_words_in_page(doctr_documents)
        stop_words = get_stop_words('french')
        stop_words.append('nan')
        for word in ['ne','le','nom','nommé','nommée','nommés']:
            try:
                stop_words.remove(word)
            except:
                print(f'{word} not in stop words already')
        self.vectorizer = CountVectorizer(strip_accents='ascii', min_df=min_df, stop_words=stop_words)
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
        array_angles = np.zeros(len(list_plain_words_in_page))  # false value for angle
        array_distances = np.ones(len(list_plain_words_in_page)) * 1  # max distance
        if vocab_i in list_plain_words_in_page:
            wi_list = [i for i, x in enumerate(list_plain_words_in_page) if x == vocab_i]
            for wi in wi_list:
                for j, word_j in enumerate(list_plain_words_in_page):
                    x_i, y_i = df.iloc[wi]['min_x'], df.iloc[wi]['min_y']
                    x_j, y_j = df.iloc[j]['min_x'], df.iloc[j]['min_y']
                    distance = cosine((x_i, y_i), (x_j, y_j))
                    if distance < array_distances[j]:  # in case there are several identical duplicate of vocab i, take the closest
                        array_angles[j] = np.arctan2((y_j - y_i), (x_j - x_i))
                        array_distances[j] = distance
        return array_angles, array_distances

    def _transform(self, X: pd.DataFrame, **kwargs):
        print(f"Transforming {self.__class__.__name__}")
        print(f"vocab that will be used for transform {self.vocab}")
        list_array_angle = []
        list_array_distance = []
        if len(X['document_name'].unique()) > 1: #this is for training or testing
            nb_cpu = mp.cpu_count()
        else: #this is for the inference
            nb_cpu = 1 #limit the number of cpu to 1 to avoid memory issues in production
        print(f'Number of CPU used by {self.__class__.__name__}: {nb_cpu}')
        with mp.Pool(nb_cpu) as pool:
            for doc in tqdm(X['document_name'].unique()):
                for page_id, page in enumerate(X[X['document_name'] == doc]['page_id'].unique()):
                    df = X[(X['document_name'] == doc) & (X['page_id'] == page)]
                    list_plain_words_in_page = self.get_words_in_page(df)
                    list_token = self.vectorizer.inverse_transform(self.vectorizer.transform(list_plain_words_in_page))
                    for i, token in enumerate(list_token):
                        if len(token) > 0:
                            list_plain_words_in_page[i] = token[0]
                    self._list_words_in_page.extend(list_plain_words_in_page)
                    vocab = self.vocab

                    res = pool.map(partial(self.get_relative_positions,list_plain_words_in_page=list_plain_words_in_page, df=df), vocab)
                    array_angle = np.array([r[0] for r in res]).T
                    array_distance = np.array([r[0] for r in res]).T
                    list_array_angle.extend(array_angle)
                    list_array_distance.extend(array_distance)
        array_angle = np.vstack(list_array_angle)
        array_distances = np.vstack(list_array_distance)
        array = np.concatenate([array_angle.T, array_distances.T])
        self.feature_names_ = [str(a) + '_angle' for a in self.vocab] + [str(a) + '_distance' for a in self.vocab]

        # TODO : keep the name of the doc and pages in a self.list_doc self.list_pages
        return array.T


    def transform(self, X: List[Document]):
        return self._transform(X)

    def _get_doctr_docs(self, raw_documents: List[Path]):
        list_doctr_docs = []
        if not hasattr(self, "doctr_model"):
            self.doctr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

        for doc in raw_documents:
            if not doc.exists():
                print(f"Doc {doc} could not be found.")
                continue
            res_doctr = None
            try:
                if doc.suffix == "pdf":
                    doc_doctr = DocumentFile.from_pdf(doc)
                else:
                    doc_doctr = DocumentFile.from_images(doc)
                res_doctr = self.doctr_model(doc_doctr, training=False)
            except Exception as e:
                print(f"Could not analyze document {doc}. Error: {e}")
            if res_doctr:
                list_doctr_docs.append(res_doctr)
        return list_doctr_docs


    @staticmethod
    def _get_neighbors_features(word, neighbors: Dict[str, List[Word]]):
        features = {}
        left_side_words = neighbors["left"]
        right_side_words = neighbors["right"]
        for id_word, word_neighbor in enumerate(left_side_words):
            features[f"w{id_word - len(left_side_words)}:{word_neighbor.value.lower()}"] = 1

        features[f"w:{word.value.lower()}"] = 1

        for id_word, word_neighbor in enumerate(right_side_words):
            features[f"w+{id_word + 1}:{word_neighbor.value.lower()}"] = 1

        return features


class ParallelWordTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def func(self, x):
        return x

    def get_feature_names(self):
        return [self.__class__.__name__]

    def transform(self, X):
        # pandarallel.initialize(progress_bar=True)
        print(f"Transforming with {self.__class__.__name__}")
        array = X['word'].to_numpy()
        f = np.vectorize(self.func)
        res = np.stack([f(array)], axis=1)
        print('Done ! ')
        return res


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

class ContainsDigit(ParallelWordTransformer):
    """
    Check if a string contains digits
    """
    def func(self, x):
        return len(re.findall(r"\d", str(x))) > 0


class IsPrenom(ParallelWordTransformer):
    """
    Check if a string is a prenom. The list of prenom is the french prenom from INSEE dataset
    """
    def __init__(self):
        super().__init__()
        with open('src/salaire/prenoms_fr_1900_2020.txt', 'r', encoding='UTf_8') as f:
            self.prenom_list = [line.strip().lower() for line in f.readlines()]

    def func(self, x):
        try:
            return str(x).lower() in self.prenom_list
        except:
            return False


class IsNom (ParallelWordTransformer):
    """
    Check if a string is a nom. The list of nom is the french nom from INSEE dataset
    """

    def __init__(self):
        super().__init__()
        with open('src/salaire/noms.txt', 'r') as f:
            self.nom_list = [line.strip().lower() for line in f.readlines()]


    def func(self, x):
        try:
            return str(x).lower() in self.nom_list
        except:
            return False


class IsDate (ParallelWordTransformer):
    def func(self, x):
        if len(re.findall(r"\d", str(x))) > 0: #check  only if there are digits otherwise it takes too long
            return dateparser.parse(str(x).lower()) is not None
        else:
            return False


class BagOfWordInLine(XtractVectorizer):
    """
    This vectorizer clusters the document by lines with Jenks Natural Breaks.
    It then finds all the words contained in a line a calculate a vector with CountVectorizer
    The vector gets calculated for every word in the document
    """

    def goodness_of_variance_fit(self, array, classes):
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

    def _transform(self, doct_documents: pd.DataFrame, **kwargs):
        print(f"Transforming {self.__class__.__name__}")
        print(f"vocab that will be used for transform {self.vocab}")

        self.array_lines = np.zeros(doct_documents.shape[0])
        self.array_bows = np.zeros((doct_documents.shape[0],len(self.vectorizer.get_feature_names())))
        i = 0
        for doc in tqdm(doct_documents['document_name'].unique()):
            for page_id, page in enumerate(doct_documents[doct_documents['document_name'] == doc]['page_id'].unique()):
                df = doct_documents[(doct_documents['document_name'] == doc) & (doct_documents['page_id'] == page)].copy()

                y = df['max_y']
                jnb = JenksNaturalBreaks()
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
        self.feature_names_ = ['lines'] +[f'bows_{f}' for f in self.vectorizer.get_feature_names()]

        # TODO : keep the name of the doc and pages in a self.list_doc self.list_pages
        return np.transpose(self.array)

    def transform(self, X: pd.DataFrame):
        return self._transform(X)






def extract_words(doctr_result: dict):
    words_dict = []
    for page in doctr_result["pages"]:
        words_dict.append(page["blocks"][0]["lines"][0]["words"])

    return words_dict


def get_doctr_info(img_path: Path) -> Document:
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    doc = DocumentFile.from_images(img_path)
    result = model(doc, training=False)
    # result.show(doc)
    return result

def get_list_words_in_page(page: Document):
    list_words_in_page = []
    for block in page.blocks:
        for line in block.lines:
            list_words_in_page.extend(line.words)
    return list_words_in_page