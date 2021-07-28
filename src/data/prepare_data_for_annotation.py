"""
Runs a baseline info extraction system. The procedure is as follows:
1. Extract boxes with doctr
2. For each word, gather the following features:
    a. min_x, min_y, max_x, max_y coordinates
    b. spacy embedding
    c. is_in_name_list
    d. is_in_prenom_list
3. Save them in a csv
"""

from pathlib import Path

from src.salaire.annotation_utils import DoctrTransformer, AnnotationDatasetCreator

IMG_FOLDER_PATH = Path("./data/CNI_recto_aligned_linux/train")
TRAINING_DATA_PATH = Path("./data/CNI_recto_aligned_linux/cni_annotation_recto_train.csv")


def main():
    list_img_paths = list([f for f in IMG_FOLDER_PATH.iterdir() if f.suffix in [".jpg", ".png"]])
    doctr_documents = DoctrTransformer().transform(list_img_paths)
    dataset_creator = AnnotationDatasetCreator(output_path=TRAINING_DATA_PATH, raw_documents=list_img_paths)
    dataset_creator.transform(doctr_documents)


if __name__ == '__main__':
    main()
