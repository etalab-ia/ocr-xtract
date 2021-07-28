from collections import Counter
from pathlib import Path
from typing import Optional, List

import pandas as pd
import json
from doctr.models import ocr_predictor
from doctr.documents import DocumentFile, Document

from src.salaire.doctr_utils import get_list_words_in_page


class DoctrTransformer:
    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, raw_documents):
        doctr_documents = self._get_doctr_docs(raw_documents=raw_documents)
        return doctr_documents

    def _get_doctr_docs(self, raw_documents: List[Path]):
        if not hasattr(self, "doctr_model"):
            self.doctr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
        list_doctr_docs = []
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


class AnnotationDatasetCreator:
    def __init__(self, output_path: Path = None, raw_documents: Optional[List[Path]] = None, sep: str = "\t"):
        self.output_path = output_path
        self.raw_documents = raw_documents
        self.sep = sep

    def fit(self, doctr_documents: List[Path], **kwargs):
        return self

    def transform(self, doctr_documents: List[Document]):
        doc_name_fake = 0
        list_lines_dicts = []
        for doc_id, doc in enumerate(doctr_documents):
            doc_name_fake += 1
            for page_id, page in enumerate(doc.pages):
                list_words_in_page = get_list_words_in_page(page)
                for word in list_words_in_page:

                    dict_info = {
                        "word": word.value,
                        "min_x": word.geometry[0][0],
                        "min_y": word.geometry[0][1],
                        "max_x": word.geometry[1][0],
                        "max_y": word.geometry[1][1],
                        "page_id": page_id
                    }
                    if self.raw_documents:
                        doc_name = self.raw_documents[doc_id].name
                        dict_info.update({"document_name": doc_name})
                    else:
                        dict_info.update({"document_name": doc_name_fake})
                    dict_info["label"] = "O"
                    list_lines_dicts.append(dict_info)
        annotation_df = pd.DataFrame(list_lines_dicts)
        if self.output_path is not None:
            annotation_df.to_csv(self.output_path, index=False, sep=self.sep)
        return annotation_df


class AnnotationJsonCreator:
    """class for generating json files in the LabelStudio json format containing the bboxes from doctr"""
    def __init__(self, raw_documents: List[Path], output_path: Path = None):
        self.output_path = output_path
        self.raw_documents = raw_documents

    def fit(self, doctr_documents: List[Path], **kwargs):
        return self

    def transform(self, doctr_documents: List[Document]):
        annotations = []
        for doc_id, doc in enumerate(doctr_documents):
            page = doc.pages[0] # On ne traite que des png/jpg donc que des docs à une page
            dict_image = {"data": {"image" : "/data/upload/{}".format(str(self.raw_documents[doc_id]).split("/")[-1])},
                          "predictions": [{'result': [], 'score': None}]} # result: list de dict pour chaque BBox

            list_words_in_page = get_list_words_in_page(page)
            height, width = page.dimensions[0], page.dimensions[1]
            id_annotation = 0
            for word in list_words_in_page:
                id_annotation += 1
                label = word.value
                xmin, ymin = word.geometry[0][0], word.geometry[0][1]
                xmax, ymax =  word.geometry[1][0],  word.geometry[1][1]
                width_a, height_a = xmax - xmin, ymax - ymin
                dict_annotation = {'id': 'result{}'.format(id_annotation),
                                   "meta": {"text": [label]},
                                   'type': 'rectanglelabels', 'from_name': 'label', 'to_name': 'image',
                                   'original_width': width, 'original_height': height, 'image_rotation': 0,
                                   'value': {'rotation': 0, 'x': xmin*100, 'y': ymin*100,
                                          'width': width_a*100, 'height': height_a*100,
                                          'rectanglelabels': [None]}}
                dict_image["predictions"][0]["result"].append(dict_annotation)

            annotations.append(dict_image)

        if self.output_path is not None:
            with open(self.output_path, 'w') as fp:
                json.dump(annotations, fp)

        return annotations

class LabelStudioConvertor:
    """class for converting label studio json files into dataframe"""

    def __init__(self, jsonfile: Path, output_path: Path = None):
        self.jsonfile = jsonfile
        self.output_path = output_path

    def transform(self, all_columns: bool = False, sep: str = "\t"):
        self.sep = sep
        data_file = open(self.jsonfile)
        data = json.load(data_file)

        df_annotations = pd.DataFrame()
        df = pd.json_normalize(data)
        df_col = ["file_upload", "created_at", "updated_at", "project", "data.image"]
        for index, row in df.iterrows():
            df_temp = pd.json_normalize(row["annotations"])
            df_temp_col = [x for x in df_temp.columns if "result" not in x]
            for index2, row2 in df_temp.iterrows():

                df2temp = pd.json_normalize(row2["result"])
                df_annotations = pd.concat([df_annotations, df2temp])
                for col in df_col:
                    df_annotations[col] = df._get_value(index, col)
                for col in df_temp_col:
                    df_annotations[col] = df_temp._get_value(index2, col)

        df_annotations["label"] = df_annotations["value.rectanglelabels"].map(
            lambda x: x[0] if pd.isna(x) == False else x)

        # rename col names
        dict_rename = {'value.x': 'min_x', 'value.y': 'min_y', 'value.rectanglelabels': 'label'}
        df_annotations.rename(columns=dict_rename, inplace=True)

        # clean columns
        df_annotations["word"] = df_annotations["meta.text"].apply(lambda x: x[0] if isinstance(x, list) else x)
        df_annotations["page_id"] = 0
        df_annotations["max_x"] = df_annotations["min_x"] + df_annotations["value.width"]
        df_annotations["max_y"] = df_annotations["min_x"] + df_annotations["value.width"]
        df_annotations["document_name"] = df_annotations["data.image"].apply(lambda x: x.split("/")[-1])

        # keep only minimal columns
        if all_columns == False:
            minimal_col_list = ['word', 'min_x', 'min_y', 'max_x', 'max_y', 'page_id', 'document_name', 'label',
                                "completed_by.email"]
            df_annotations = df_annotations[minimal_col_list]

        if self.output_path is not None:
            df_annotations.to_csv(self.output_path, index=False, sep=self.sep)

        return df_annotations

