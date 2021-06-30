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

from pathlib import Path
from pprint import pprint

from doctr.models import ocr_predictor
from doctr.documents import DocumentFile

from doctr_utils import get_doctr_info, extract_words, WindowTransformer

IMG_PATH = "/data/dossierfacil/salary/notvalidated_png/68945ca2-1758-48d0-b2bd-4ebb056fa752.pdf-1.png"
IMG_PATH = "data/CNI_caro2.jpg"
IMG_PATH = "data/476922b7-0bdf-414c-a7ef-6c1a0c3618c9.jpg"

def create_windows(horizontal: int = 3, vertical: int = 3):
    pass


def main(image_path: Path):
    doct_output = get_doctr_info(image_path) for image_path in list
    # doct_output.pages[0].blocks[0].lines[0].words
    windower = WindowTransformer(line_eps=0.01)
    windower.fit([doct_output])
    windower._transform([doct_output])
    # windower.get_sourrounding_words(id_box=10)
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main(IMG_PATH)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
