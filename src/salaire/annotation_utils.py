from collections import Counter
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from doctr.models import ocr_predictor
from doctr.documents import DocumentFile, Page, Document, Word
from sklearn.feature_extraction import DictVectorizer
from scipy.spatial.distance import cosine

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
    def __init__(self, output_path: Path, raw_documents: Optional[List[Path]] = None):
        self.output_path = output_path
        self.raw_documents = raw_documents

    def fit(self, doctr_documents: List[Path], **kwargs):
        return self

    def transform(self, doctr_documents: List[Document]):

        list_lines_dicts = []
        for doc_id, doc in enumerate(doctr_documents):
            for page_id, page in enumerate(doc.pages):
                list_words_in_page = get_list_words_in_page(page)
                for word in list_words_in_page:

                    dict_info = {
                        "word": word.value,
                        "min_x": word.geometry[0][0],
                        "min_y": word.geometry[0][1],
                        "max_x": word.geometry[1][1],
                        "max_y": word.geometry[1][1],
                        "page_id": page_id
                    }
                    if self.raw_documents:
                        doc_name = self.raw_documents[doc_id].name
                        dict_info.update({"document_name": doc_name})
                    dict_info["label"] = "O"
                    list_lines_dicts.append(dict_info)
        annotation_df = pd.DataFrame(list_lines_dicts)
        annotation_df.to_csv(self.output_path, index=False)
