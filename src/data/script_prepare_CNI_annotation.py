from src.salaire.annotation_utils import DoctrTransformer, AnnotationJsonCreator
from pathlib import Path
import os


img_folder_path = "data/batch_1_Robin/CNI_recto_test_aligned"
output_path = "data/batch_1_Robin/cni_annotation_test_2.json"


def main():
    list_img_path = [Path(os.path.join(img_folder_path, x)) for x in os.listdir(img_folder_path)]
    list_doctr_docs = DoctrTransformer().transform(list_img_path)
    annotations = AnnotationJsonCreator(list_img_path, output_path).transform(list_doctr_docs)

if __name__ == '__main__':
    main()
