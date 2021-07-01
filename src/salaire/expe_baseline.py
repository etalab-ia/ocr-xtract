"""
Runs a baseline info extraction system. The procedure is as follows:
1. Extract boxes with doctr
2. For each word, gather the following features:
    a. min_x, min_y, max_x, max_y coordinates
    b. spacy embedding
    c. is_in_name_list
    d. is_in_prenom_list
3. Y is one of the following classes [prenom, nom, nom_entreprise, salaire_net]
4. Train a simple classification algorithm to predict the above mentioned classes

"""
import glob
from pathlib import Path
from typing import List
from pprint import pprint
import numpy as np

from doctr.models import ocr_predictor
from doctr.documents import DocumentFile

from doctr_utils import get_doctr_info, extract_words, WindowTransformer
from src.salaire.annotation_utils import DoctrTransformer, AnnotationDatasetCreator

IMG_PATH = "/data/dossierfacil/salary/notvalidated_png/68945ca2-1758-48d0-b2bd-4ebb056fa752.pdf-1.png"
IMG_FOLDER_PATH = Path("/data/dossierfacil/CNI_recto/train")
TRAINING_DATA_PATH = Path("./data/training_data/cni_annotation_recto_train.csv")


def main():
    list_img_paths = list(sorted([f for f in IMG_FOLDER_PATH.iterdir() if f.suffix in [".jpg", ".png"]]))
    doctr_documents = DoctrTransformer().transform(list_img_paths)
    dataset_creator = AnnotationDatasetCreator(output_path=TRAINING_DATA_PATH, raw_documents=list_img_paths)
    dataset_creator.transform(doctr_documents)

    # windower = WindowTransformer(line_eps=0.01)
    # windower.fit(doctr_documents)
    # windower._transform(doctr_documents)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
