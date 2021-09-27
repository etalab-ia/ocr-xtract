from src.salaire.annotation_utils import DoctrTransformer
from pathlib import Path
import os
import json


img_folder_path = "/Users/kimmontalibet/data/AFA/test/"
output_path = "/Users/kimmontalibet/data/AFA/outputOCR/"


def main():
    list_img_path = [Path(os.path.join(img_folder_path, x)) for x in os.listdir(img_folder_path)]
    list_doctr_docs = DoctrTransformer().transform(list_img_path)
    for doc_id, doc in list_doctr_docs:
        with open(output_path + "ocr_doc_{}.json".format(doc_id), 'w') as fp:
            json.dump(doc, fp)


if __name__ == '__main__':
    main()
