import difflib
from collections import Counter
from pathlib import Path
from pprint import pprint
from typing import Optional, List, Dict
import re

import numpy as np
from doctr.models import ocr_predictor
from doctr.documents import DocumentFile, Page, Document, Word
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from stop_words import get_stop_words

import dateparser


class PageExtended(Page):
    pass


class WindowTransformerList(DictVectorizer):
    def __init__(self, searched_words: List = None):
        super().__init__(sparse=False)
        self.searched_words=searched_words
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

    def get_middle_position(self, df, i):
        return (df.iloc[i]["min_x"] + df.iloc[i]["max_x"]) / 2, (df.iloc[i]["min_y"] + df.iloc[i]["max_y"]) / 2

    def _transform(self, doct_documents: pd.DataFrame, **kwargs):
        print(f"vocab that will be used for transform {self.vocab}")
        list_array_angle = []
        list_array_distance = []
        for doc in doct_documents['document_name'].unique():
            for page_id, page in enumerate(doct_documents[doct_documents['document_name'] == doc]['page_id'].unique()):
                df = doct_documents[(doct_documents['document_name'] == doc) & (doct_documents['page_id'] == page)]
                list_plain_words_in_page = self.get_words_in_page(df)
                list_token = self.vectorizer.inverse_transform(self.vectorizer.transform(list_plain_words_in_page))
                for i, token in enumerate(list_token):
                    if len(token) > 0:
                        list_plain_words_in_page[i] = token[0]
                self._list_words_in_page.extend(list_plain_words_in_page)
                vocab = self.vocab
                array_angles = np.zeros((len(vocab), len(list_plain_words_in_page)))  # false value for angle
                array_distances = np.ones((len(vocab), len(list_plain_words_in_page))) * 1  # max distance

                for i, vocab_i in enumerate(vocab):
                    if vocab_i in list_plain_words_in_page:
                        wi_list = [i for i, x in enumerate(list_plain_words_in_page) if x == vocab_i]
                        for wi in wi_list:
                            for j, word_j in enumerate(list_plain_words_in_page):
                                x_i, y_i = df.iloc[wi]['min_x'], df.iloc[wi]['min_y']
                                x_j, y_j = df.iloc[j]['min_x'], df.iloc[j]['min_y']
                                distance = cosine((x_i, y_i), (x_j, y_j))
                                if distance < array_distances[
                                    i, j]:  # in case there are several identical duplicate of vocab i, take the closest
                                    array_angles[i, j] = np.arctan2((y_j - y_i), (x_j - x_i))
                                    array_distances[i, j] = distance
                    else:
                        print(f'--------------vocab------{vocab_i} not in page')

                list_array_angle.append(array_angles)
                list_array_distance.append(array_distances)
        self.array_angle = np.hstack(list_array_angle)
        self.array_distances = np.hstack(list_array_distance)
        self.array = np.concatenate([self.array_angle, self.array_distances])
        self._feature_names = [str(a) + '_angle' for a in self.vocab] + [str(a)  + '_distance' for a in self.vocab]

        # TODO : keep the name of the doc and pages in a self.list_doc self.list_pages
        return np.transpose(self.array)

    def fit(self, doctr_documents: pd.DataFrame, min_df=0.2, **kwargs):
        min_df = int(min_df * len(doctr_documents['document_name'].unique()))
        self.list_words = self.get_words_in_page(doctr_documents)
        stop_words = get_stop_words('french')
        stop_words.append('nan')
        for word in ['ne','le','nom','nommé','nommée','nommés']:
            stop_words.remove(word)
        self.vectorizer = CountVectorizer(strip_accents='ascii', min_df=min_df, stop_words=stop_words)
        self.vectorizer.fit(self.list_words)
        self.vocab = self.vectorizer.get_feature_names()
        return self

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

    def fit_transform(self, X: List[Document],y = None, **kwargs):
        self.fit(X)
        return self._transform(X)

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


class BoxPositionGetter(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def find_middle(self, X):
        X["middle_x"] = (X['min_x'] + X["max_x"])/2
        X["middle_y"] = (X['min_y'] + X["max_y"])/2
        return X[["middle_x", "middle_y"]]

    def transform(self, X):
        return X.apply(self.find_middle, axis=1).to_numpy()

class ContainsDigit(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def contains_digit(self, x):
        return len(re.findall(r"\d", str(x))) > 0

    def transform(self, X):
        return np.stack([X['word'].apply(lambda x: self.contains_digit(x)).to_numpy().astype(int)], axis =1)

class IsPrenom(TransformerMixin, BaseEstimator):
    def __init__(self):
        with open('src/salaire/prenoms_fr_1900_2020.txt', 'r') as f:
            self.prenom_list = [line.strip().lower() for line in f.readlines()]

    def fit(self, X, y=None):
        return self

    def is_prenom(self, x):
        return str(x).lower() in self.prenom_list

    def transform(self, X):
        return np.stack([X['word'].apply(lambda x: self.is_prenom(x)).to_numpy().astype(int)], axis=1)

class IsNom (TransformerMixin, BaseEstimator):
    def __init__(self):
        with open('src/salaire/noms.txt', 'r') as f:
            self.nom_list = [line.strip().lower() for line in f.readlines()]

    def fit(self, X, y=None):
        return self

    def is_nom(self, x):
        return str(x).lower() in self.nom_list

    def transform(self, X):
        return np.stack([X['word'].apply(lambda x: self.is_nom(x)).to_numpy().astype(int)], axis=1)

class IsDate (TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def is_date(self, x):
        if len(re.findall(r"\d", str(x))) > 0: #check  only if there are digits otherwise it takes too long
            return dateparser.parse(str(x).lower()) is not None
        else:
            return False

    def transform(self, X):
        return np.stack([X['word'].apply(lambda x: self.is_date(x)).to_numpy().astype(int)], axis=1)


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
