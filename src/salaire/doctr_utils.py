from collections import Counter
from pathlib import Path
from pprint import pprint
from typing import Optional, List, Dict

import numpy as np
from doctr.models import ocr_predictor
from doctr.documents import DocumentFile, Page, Document, Word
from sklearn.feature_extraction import DictVectorizer
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer


class PageExtended(Page):
    pass


class WindowTransformer(DictVectorizer):
    def __init__(self, horizontal: int = 3, vertical: int = 3, page_id: Optional[int] = None, line_eps: float = 2):

        # self.document = doctr_doc
        # self.document.pages[0].blocks[0].lines[0].words
        super().__init__(sparse=False)
        self.neighbors = []
        self.n_horizontal = horizontal
        self.n_vertical = vertical
        self.coordinates_sorted_ids = []
        self.page_id = page_id
        self.line_eps = line_eps
        self._features = None
        self._list_same_line_arr = None
        self._list_words_per_page = None

    def _get_neighbors(self, target_word_idx: int, target_word: Word, page_id: int):
        same_line_indices = self._list_same_line_arr[page_id][target_word_idx].nonzero()[0]
        same_line_words = list(self._list_words_per_page[page_id][same_line_indices])
        same_line_words.append(target_word)
        same_line_words = np.array(same_line_words)
        same_line_x_min = [w.geometry[0][0] for w in same_line_words]

        sorted_points_in_line = np.argsort(same_line_x_min)
        target_word_idx_sorted = np.where(sorted_points_in_line == len(same_line_x_min) - 1)[0][0]
        left_neighbor_points_indices = sorted_points_in_line[
                                       max(0, target_word_idx_sorted - self.n_horizontal):target_word_idx_sorted]
        right_neighbor_points_indices = sorted_points_in_line[
                                        target_word_idx_sorted + 1: target_word_idx_sorted + self.n_horizontal]
        left_neighbor_words = same_line_words[left_neighbor_points_indices]
        right_neighbor_words = same_line_words[right_neighbor_points_indices]
        return {"left": left_neighbor_words, "right": right_neighbor_words}

    def _transform(self, X: List[Document], **kwargs):
        # todo check X is list or not
        list_array_angle = []
        list_array_distance = []
        for doc in X:
            for page_id, page in enumerate(doc.pages):
                list_words_in_page = []
                for block in page.blocks:
                    for line in block.lines:
                        list_words_in_page.extend(line.words)

                vocab = self.vocab
                array_angles = np.ones((len(vocab),len(list_words_in_page))) * 5 # false value for angle
                array_distances = np.zeros((len(vocab),len(list_words_in_page))) # max distance
                for i, word_i in enumerate(vocab):
                    if word_i in [word.value for word in list_words_in_page]:
                        wi = [word.value for word in list_words_in_page].index(word_i)
                        word_i = list_words_in_page[wi]
                        for j, word_j in enumerate(list_words_in_page):
                            x_i, y_i = word_i.geometry[0]
                            x_j, y_j = word_j.geometry[0]
                            array_angles[i,j] = np.arctan((y_j-y_i)/(x_j-x_i) if (x_j-x_i) !=0 else 0)
                            array_distances[i,j] = cosine(word_i.geometry[0], word_j.geometry[0])
                    else:
                        print(f'--------------------{word_i}')
                        print([word.value for word in list_words_in_page])

                list_array_angle.append(array_angles)
                list_array_distance.append(array_distances)
        self.array_angle = np.hstack(list_array_angle)
        self.array_distances = np.hstack(list_array_distance)

        return self

    def fit(self, X: List[Document], **kwargs):
        # self._get_sorted_coordinates(X)
        # get ALL Words of a page
        # TODO: Treat more carefully the blocks and lines

        # TODO check if X is [] or Doc : if doc, then [doc]

        list_words = []
        for doc in X:
            for page_id, page in enumerate(doc.pages):
                for block in page.blocks:
                    for line in block.lines:
                        list_words.extend([word.value for word in line.words])

        # vectorizer = CountVectorizer(min_df=1)
        # vectorizer.fit(list_words)
        #
        # self.vectorizer = vectorizer
        # self.vocab = vectorizer.get_feature_names()
        self.vocab = [k for k, v in Counter(list_words).items() if v >= 1]
        return self

    def transform(self, X):
        return self._transform(X)

    def fit_transform(self, X: List[Page], **kwargs):
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
