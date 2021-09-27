from pathlib import Path
import os
import json
from doctr.documents import DocumentFile
from doctr.models import ocr_predictor

img_folder_path = "/Users/kimmontalibet/data/AFA/Bastia/"
output_path = "/Users/kimmontalibet/data/AFA/outputOCR/"



def main():
    list_img_path = [Path(os.path.join(img_folder_path, x)) for x in os.listdir(img_folder_path)]
    model = ocr_predictor(pretrained=True)
    for doc_id, pathpdf in enumerate(list_img_path):
        doc = DocumentFile.from_pdf(pathpdf).as_images()
        result = model(doc)
        json_output = result.export()

        with open(output_path + "ocr_doc_{}.json".format(doc_id), 'w') as fp:
            json.dump(json_output, fp)


if __name__ == '__main__':
    main()
