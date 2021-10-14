import math
import random

import cv2
import numpy as np
import pandas as pd

from doctr.utils.geometry import rotate_abs_boxes, rbbox_to_polygon, rotate_boxes, polygon_to_bbox
from doctr.utils import compute_expanded_shape
from sklearn.base import TransformerMixin, BaseEstimator

# rotate page : OK
# resize page
# move boxes
# resize boxes
# mix images
# random erasing


def bbox_width_to_bbox(bbox):
    return bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]


class AugmentDocuments (TransformerMixin, BaseEstimator):
    def __init__(self, list_processes=None):
        if list_processes is None:
            list_processes = ['rotate_page']
        self.list_processes = list_processes

    def fit(self, X, y=None):
        return self

    def get_rel_boxes(self, df: pd.DataFrame):
        min_x = (df['min_x']).to_list()
        min_y = (df['min_y']).to_list()
        max_x = (df['max_x']).to_list()
        max_y = (df['max_y']).to_list()
        return np.array([[mi_x, mi_y, ma_x, ma_y] for mi_x, mi_y, ma_x, ma_y in zip(min_x, min_y, max_x, max_y)])

    def replace_boxes(self, df, boxes):
        df['min_x'] = boxes[:, 0]
        df['min_y'] = boxes[:, 1]
        df['max_x'] = boxes[:, 2]
        df['max_y'] = boxes[:, 3]
        return df

    def rotate_doc (self, X, doc, angles=None):
        if angles is None:
            angles = [-1., -0.5, 0.5, 1.]
        list_df = []
        for page_id, page in enumerate(X[X['document_name'] == doc]['page_id'].unique()):
            for angle in angles:
                df = X[(X['document_name'] == doc) & (X['page_id'] == page)].copy()
                boxes = self.get_rel_boxes(df)
                r_boxes = rotate_boxes(boxes, angle=angle, min_angle=0)
                # the multiplications by 1000 is to get some more details for cv2.boundingRect
                r_poly = np.array([cv2.boundingRect(rbbox_to_polygon(rbox)*1000) for rbox in r_boxes])/1000
                boxes_tilted = np.array([bbox_width_to_bbox(r_p) for r_p in r_poly])
                df['document_name'] = df['document_name'] + f'_angle_{angle}'
                df = self.replace_boxes(df, boxes_tilted)
                list_df.append(df)
        return list_df

    def random_mix(self, X, random_state=42):
        # TODO this might induce overfitting : evalutate before usage by default
        list_df = []
        X['doc_pages'] = X.apply(lambda x: str(x['document_name']) + str(x['page_id']), axis=1)
        list_pages = X['doc_pages'].unique()
        number_of_pages = len(list_pages)
        random.seed(random_state)
        list_pages_top = random.sample(range(number_of_pages), number_of_pages)
        random.seed(random_state + 1)
        list_pages_down = random.sample(range(number_of_pages), number_of_pages)
        for i, j in zip(list_pages_top, list_pages_down):
            df_top = X[(X['doc_pages'] == list_pages[i]) & (X['max_y'] < 0.5)].copy()
            df_bottom = X[(X['doc_pages'] == list_pages[j]) & (X['max_y'] >= 0.5)].copy()
            df = pd.concat([df_bottom, df_top]).drop(columns=['doc_pages'], inplace=False)
            df['document_name'] = df['document_name'] + f'_split_{i}_{j}'
            list_df.append(df)
        X.drop(columns=['doc_pages'], inplace=True)
        return list_df

    def transform(self, X):
        print(f"Transforming with {self.__class__.__name__}")
        list_df = [X]
        if "mix_images" in self.list_processes:
            list_df.extend(self.random_mix(X, random_state=42))
        if "rotate_page" in self.list_processes:
            for doc in X['document_name'].unique():
                list_df.extend(self.rotate_doc(X, doc))
        return pd.concat(list_df)


if __name__ == "__main__":

    data_train = pd.read_csv("./data/salary_for_training/train_annotated.csv", sep='\t')

    aug = AugmentDocuments()
    data_train = aug.transform(data_train)

    columns = data_train.columns.to_list()
    columns.remove('label')
    X_train, y_train = data_train[columns], data_train["label"]

    X_train = aug.transform(X_train)