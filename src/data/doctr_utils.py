from pathlib import Path
from typing import List, Optional

import pandas as pd
from doctr.io import Document, DocumentFile
from doctr.models import ocr_predictor


def get_list_words_in_page(page: Document):
    list_words_in_page = []
    for block in page.blocks:
        for line in block.lines:
            list_words_in_page.extend(line.words)
    return list_words_in_page


class DoctrTransformer:
    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, raw_documents, rotate_document: bool= False):
        doctr_documents = self._get_doctr_docs(raw_documents=raw_documents, rotate_document=rotate_document)
        return doctr_documents

    def _get_doctr_docs(self, raw_documents: List[Path], rotate_document: bool):
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
                res_doctr = self.doctr_model(doc_doctr, rotate_document=rotate_document)
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