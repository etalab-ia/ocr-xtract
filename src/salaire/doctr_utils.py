from pathlib import Path
from pprint import pprint
from typing import Optional, List

import numpy as np
from doctr.models import ocr_predictor
from doctr.documents import DocumentFile, Page, Document, Word


class PageExtended(Page):
    pass


class WindowTransformer:
    def __init__(self, horizontal: int = 3, vertical: int = 3, page_id: Optional[int] = None,
                 line_threshold: int = 2):

        # self.document = doctr_doc
        # self.document.pages[0].blocks[0].lines[0].words
        self.neighbors = []
        self.n_horizontal = horizontal
        self.n_vertical = vertical
        self.coordinates_sorted_ids = []
        self.page_id = page_id
        self.line_threshold = line_threshold
        self._features = None

        pass

    def _get_sorted_coordinates(self, document):
        x_min, y_min = [], []
        for pages in document:
            for blocks in pages.blocks:
                for lines in blocks.lines:
                    for word in lines.words:
                        x_min.append(word.geometry[0][0])
                        y_min.append(word.geometry[0][0])
            self.coordinates_sorted_ids.append([np.array(x_min).argsort(),
                                                np.array(y_min).argsort()])

    def _get_neighbors(self, target_word_idx: int, page_id: int, list_words: List[Word]):
        sorted_x_min = self.coordinates_sorted_ids[page_id][0]
        sorted_y_min = self.coordinates_sorted_ids[page_id][1]
        target_word: Word = list_words[target_word_idx]
        target_word_x_min = target_word.geometry[0][0]
        target_word_y_min = target_word.geometry[0][1]
        list_words.pop(target_word_idx)  # remove this word from the list
        for word_id, word in enumerate(list_words):
            current_word_x = word.geometry[0][0]
            current_word_y = word.geometry[0][1]
            if target_word_x_min == current_word_x and target_word_y_min == current_word_y:
                continue
            # we are in the same line within a given threshold (line_threshold)
            if target_word_y_min - self.line_threshold < current_word_y < target_word_y_min + self.line_threshold:
                # now we want to find the n_horizontal neighbors to the left
                sorted_id = np.where(sorted_x_min == word_id)[0][0]
                left_neighbors_ids = max(0, sorted_id - self.n_vertical)
                right_neighbors_ids = min(len(list_words) - 1, sorted_id - self.n_horizontal)
                pass
            else:
                continue

    def _transform(self, X, fitting=False):
        if fitting:
            self._get_sorted_coordinates(X)
            pass

        if self.page_id:
            X = [X[self.page_id]]

        min_x, min_y = [], []

        for page_id, pages in enumerate(X):
            for blocks in pages.blocks:
                for lines in blocks.lines:
                    for word_idx, word in enumerate(lines.words):
                        self._get_neighbors(word_idx, page_id, lines.words)
            #             min_x.append(word["geometry"][0][0])
            #             min_y.append(word["geometry"][0][0])
            # self.coordinates_sorted_ids.append([np.array(min_x).argsort(),
            #                                     np.array(min_y).argsort()])

        #
        # sorted_box_x_idx = np.where(self.sorted_min_x == box_id)[0]
        # sorted_box_y_idx = np.where(self.sorted_min_y == box_id)[0]
        #
        # window_left = max(0, sorted_box_x_idx - horizontal)
        # window_right = sorted_box_x_idx + horizontal
        # window_top = max(0, sorted_box_y_idx - vertical)
        # window_bottom = sorted_box_y_idx + vertical

    def fit(self, X: Document):
        self._get_sorted_coordinates(X)
        return self

    def transform(self, X):
        return self._transform(X, fitting=False)

    def fit_transform(self, X: List[Page]):
        return self._transform(X, fitting=True)


def extract_words(doctr_result: dict):
    words_dict = []
    for page in doctr_result["pages"]:
        words_dict.append(page["blocks"][0]["lines"][0]["words"])

    return words_dict


def get_ordered_words(words_dict):
    x_min_sorted, x_max_sorted, y_min_sorted = [], [], [], []


def get_doctr_info(img_path: Path) -> Document:
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    doc = DocumentFile.from_images(img_path)
    result = model(doc)
    # result.show(doc)
    return result
