"""
metrics
f1 score per field
mean f1 score
"""

import json
from collections import Counter
from pathlib import Path

from src.image.image import RectoCNI


def calculate_f1(OCR, annotation):
    def shared_chars(s1, s2):
        return sum((Counter(s1) & Counter(s2)).values())

    sum_quality =0
    nb_zones =0

    for field in OCR.keys():
        if field not in annotation.keys():
            continue
        else:
            print(OCR[field]['field'] +", " +annotation[field])
            true_positives = shared_chars(OCR[field]['field'], annotation[field])
            false_positives = len(OCR[field]['field']) - true_positives
            false_negatives = len(annotation[field]) - true_positives
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            try:
                f1 = 2 * (precision * recall) / (precision + recall)
            except:
                f1 = 0
            sum_quality += f1
            nb_zones += 1
    return sum_quality / nb_zones

def check_doc_type(image, type):
    """

    :param image:
    :param type: the type annotated
    :return:
    """
    for annotation in image["annotations"][0]["result"]:
        if annotation["from_name"] == "R/V" and annotation["value"]["text"][0].lower() == type.lower():
            return True
    return False


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


if __name__ == "__main__":
    with open(Path('./data/annotation_results.json'), 'r', encoding='UTF_8') as file:
        data = json.load(file)
    sum_f1 = 0
    nb_f1 = 0

    for image_annotated in data:
        conditions = check_doc_type(image_annotated,'recto')
        annotated_image_path = Path('./data/'+image_annotated["file_upload"])
        conditions = conditions and annotated_image_path.exists() # check with image exists
        if conditions:
            image_ocr = RectoCNI(annotated_image_path)
            OCR = image_ocr.extract_ocr()
            annotation = extract_annotation(image_annotated)
            sum_f1 += calculate_f1(OCR, annotation)
            nb_f1 += 1

    print(sum_f1/nb_f1)