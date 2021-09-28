from src.data.annotation_utils import AnnotationJsonCreator
from src.data.doctr_utils import DoctrTransformer
from pathlib import Path
import os


img_folder_path = "data/salary/train_kim"
output_path = "data/salary/train_kim.json"


def main():
    list_img_path = [Path(os.path.join(img_folder_path, x)) for x in os.listdir(img_folder_path)]
    list_doctr_docs = DoctrTransformer().transform(list_img_path)
    annotations = AnnotationJsonCreator(list_img_path, output_path).transform(list_doctr_docs)

if __name__ == '__main__':
    main()
