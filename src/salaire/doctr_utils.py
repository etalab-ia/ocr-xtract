from pathlib import Path
from pprint import pprint
from typing import Optional, List, Dict

import numpy as np
from doctr.models import ocr_predictor
from doctr.documents import DocumentFile, Page, Document, Word
from sklearn.feature_extraction import DictVectorizer


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

    def _transform(self, X, **kwargs):
        self.fit(X)

        if self.page_id:
            self._list_words_per_page = [self._list_words_per_page[self.page_id]]
            self._list_same_line_arr = [self._list_same_line_arr[self.page_id]]

        features_dicts = []

        for page_id, page in enumerate(self._list_words_per_page):
            for word_idx, word in enumerate(page):
                neighbors = self._get_neighbors(word_idx, word, page_id)
                features = self._get_neighbors_features(word, neighbors)
                features_dicts.append(features)
            X_matrix = super().fit_transform(features_dicts)
            break  # TODO: in fact i should not deal with multiple pages :(

        return X_matrix

    def fit(self, X: List[Document], **kwargs):
        # self._get_sorted_coordinates(X)
        # get ALL Words of a page
        # TODO: Treat more carefully the blocks and lines
        list_words_per_page = []
        for page_id, page in enumerate(X):
            list_words = []
            for block in page.blocks:
                for line in block.lines:
                    list_words.extend(line.words)
            list_words_per_page.append(np.array(list_words))

        list_same_line_arr = []
        for page_id, words in enumerate(list_words_per_page):
            is_in_line_arr = np.zeros((len(words),) * 2)
            for i, word_i in enumerate(words[:-1]):
                for j, word_j in enumerate(words[i + 1:]):
                    if abs(word_j.geometry[0][1] - word_i.geometry[0][1]) < self.line_eps:
                        is_in_line_arr[i, j + 1] = 1

            list_same_line_arr.append(is_in_line_arr)
            assert is_in_line_arr.shape[0] == len(words)

        self._list_same_line_arr = list_same_line_arr
        self._list_words_per_page = list_words_per_page

        # vocabulary

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
    result = model(doc)
    # result.show(doc)
    return result
