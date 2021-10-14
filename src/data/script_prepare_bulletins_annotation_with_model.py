from src.data.annotation_utils import AnnotationJsonCreator
from src.data.doctr_utils import DoctrTransformer, AnnotationDatasetCreator
from pathlib import Path
import os
import pickle

img_folder_path = "data/bulletins"
output_path = "data/bulletins/label_studio_test.json"
model_path = "./model/fdp_model_automl"

# Write here if you wish to create json file for uploaded images (upoload = True)
# or for loaded with docker (upload = False by default)
upload = True



with open("./model/fdp_model_automl", 'rb') as data_model:
    pipe_feature = pickle.load(data_model)
    classifier = pickle.load(data_model)



def main():
    list_img_path = [Path(os.path.join(img_folder_path, x)) for x in os.listdir(img_folder_path) if x.endswith((".jpg", ".png", ".jpeg"))]
    list_doctr_docs = DoctrTransformer().transform(list_img_path)
    annotations = AnnotationJsonCreator(list_img_path, output_path).transform(list_doctr_docs)
    dataset_creator = AnnotationDatasetCreator()
    df_bbox = dataset_creator.transform(list_doctr_docs)

    X_feats = pipe_feature.transform(df_bbox)
    predictions = classifier.predict(X_feats)
    annotations = AnnotationJsonCreator(list_img_path, output_path, list(predictions)).transform(list_doctr_docs, upload)


if __name__ == '__main__':
    main()
