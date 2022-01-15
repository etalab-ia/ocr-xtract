from src.data.annotation_utils import AnnotationJsonCreator
from src.data.doctr_utils import DoctrTransformer, AnnotationDatasetCreator
from pathlib import Path
import os
import pickle


img_folder_path = "data/CNI/test"
output_path = "data/CNI/label_studio_test.json"
model_path = "model/CNI_model"


# Write here if you wish to create json file for uploaded images (true) or for loaded with docker (false by default)
upload = True


model = pickle.load(open(model_path, 'rb'))


def main():
    list_img_path = [Path(os.path.join(img_folder_path, x)) for x in os.listdir(img_folder_path) if x.endswith((".jpg", ".png", ".jpeg"))]
    list_doctr_docs = DoctrTransformer().transform(list_img_path)
    annotations = AnnotationJsonCreator(list_img_path, output_path).transform(list_doctr_docs)
    dataset_creator = AnnotationDatasetCreator()
    df_bbox = dataset_creator.transform(list_doctr_docs)
    predictions = model.predict(df_bbox)
    annotations = AnnotationJsonCreator(list_img_path, output_path, list(predictions)).transform(list_doctr_docs, upload)


if __name__ == '__main__':
    main()
