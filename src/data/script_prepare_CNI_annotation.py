from src.data.annotation_utils import AnnotationJsonCreator
from src.data.doctr_utils import DoctrTransformer
from pathlib import Path
import os
from pdf2image import convert_from_path
import pandas as pd
import xlrd


path1 = "/Users/kimmontalibet/data/dossierfacil/"


img_folder_path = "data/quittances/sample5/"
output_path = "data/quittances/preannotation/quittance_preannotation_sample5.json"

img_folder_path = "data/salary/batch2"
output_path = "data/salary/batch2.json"


def main():
    list_img_path = [os.path.join(img_folder_path, x) for x in os.listdir(img_folder_path)]
    #dfs = pd.read_csv(path1 + "list_quittance_files_sample1.csv")
    #list_img_path = dfs["path"].tolist()
    # convert pdf to img and save them
    for path in list_img_path:
        if path.endswith(('.pdf', '.PDF')):
            pages = convert_from_path(path, 500)
            path_no_suffix = path[:-4]
            for page_index, page in enumerate(pages):
                page.save(path_no_suffix + "_page_{}".format(page_index) + '.jpg', 'JPEG')

    list_img_path = [Path(os.path.join(img_folder_path, x)) for x in os.listdir(img_folder_path) if x.endswith(('.jpg', '.jpeg', ".png"))]
    list_doctr_docs = DoctrTransformer().transform(list_img_path)
    annotations = AnnotationJsonCreator(list_img_path, output_path).transform(list_doctr_docs)

if __name__ == '__main__':
    main()
