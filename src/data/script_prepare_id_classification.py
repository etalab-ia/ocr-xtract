"""
This script is used for the preparation of the dataset for the training of the classification algorithm
"""


import json
import shutil
from collections import Counter
from pathlib import Path
import os


def extract_doc_type(image_annotation):
    """
    This function extracts the document type from the annotation performed on the Identity Documents
    :param image_annotation: the annotation from the annotated image
    :return: the document type
    """
    recto_verso = ""
    type = ""
    for annotation in image_annotation["annotations"][0]["result"]:
        if annotation["from_name"] == "R/V":
            recto_verso = '_'.join(annotation["value"]["text"])
        if annotation["from_name"] == "Type":
            type = '_'.join(annotation["value"]["text"]) #in case there are several annotation
    return type

# TODO add some cleaning for recto_verso

def extract_annotation(image):
    annotations = {}
    for annotation in image['annotations'][0]["result"]:
        if annotation["from_name"] == "Nom":
            annotations['nom'] = annotation["value"]["text"][0]
        if annotation["from_name"] == "Prenom":
            annotations['prenom'] = annotation["value"]["text"][0]
        if annotation["from_name"] == "Date de Naissance":
            annotations['date_de_naissance'] = annotation["value"]["text"][0]
    return annotations


def describe_annotation_file(data):
    number_of_image = 0
    fields_annotated = set()
    for image_annotation in data:
        number_of_image += 1
        for annotation in image_annotation['annotations']:
            for field in annotation['result']:
                fields_annotated.add(field['from_name'])
    print(f'Number of image annotated in data  : {number_of_image}')
    print(f'List of annotation fields : {fields_annotated}')


def save_image(input_folder, output_folder, image_annotation, dataset, category):
    original_file_path = input_folder / image_annotation["file_upload"]
    constructed_file_path = output_folder / dataset/ category / (category + '_' + image_annotation["file_upload"])
    os.makedirs(os.path.dirname(constructed_file_path), exist_ok=True)
    shutil.copy(original_file_path, constructed_file_path)
    pass


if __name__ == "__main__":
    input_folder = Path("/root/dossierfacil/CNI/validated")
    annotation_file = Path('./data/annotation_results.json')
    output_folder = Path('./data/annotation_ID')

    with open(annotation_file, 'r', encoding='UTF_8') as file:
        data = json.load(file)

    describe_annotation_file(data)

    for image_annotation in data:
        category = extract_doc_type(image_annotation)
        dataset = 'train' #we should also add validation and test
        annotated_image_path = Path('./data/' + image_annotation["file_upload"])
        if annotated_image_path.exists():
            save_image(input_folder, output_folder, image_annotation, dataset, category)