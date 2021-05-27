import numpy as np
import imutils
import cv2

from src.image.image import Image, RectoCNI, VersoCNI
from src.image.preprocessing import align_images
from src.image.remove_noise import ImageCleaning

class Document():
    def __init__(self):
        self.images = {}

    def add_image (self, image_path, type=None):
        if type and type not in self.images.keys():
            self.images[type] = []
        if type == 'RectoCNI':
            self.images[type].append(RectoCNI(image_path))
        elif type == 'VersoCNI':
            self.images[type].append(VersoCNI(image_path))
        elif type == None:
            # TODO add some logic to use classify image
            self.images['unknown'] = Image(image_path)

    def classify_image (self, image_path):
        type = None
        print(f'Image classified as type {type}')
        return type

    def align_images(self, debug=False):
        for image_type in self.images.keys():
            if image_type == 'unknown':
                continue
            for image in self.images[image_type]:
                image.align_images(debug)

    def clean_images(self):
        for image_type in self.images.keys():
            for image in self.images[image_type]:
                image.clean()

    def extract_ocr(self, debug=False):
        for image_type in self.images.keys():
            for image in self.images[image_type]:
                image.extract_ocr(ocr_unknown_fields_only=False, debug=debug)

    def select_extracted_field(self, extracted_information):
        """
        this function only selects the fields of interests (named 'fields') from the image.extractedinformation.
        It removes the known fields named 'titles'
        :param extracted_information: the extracted information from the image
        :return: a dict with {field_name: extracted information}
        """
        selection = {}
        for information in extracted_information.keys():
            if 'field' in extracted_information[information].keys():
                selection[information] = extracted_information[information]['field']
        return selection

    def export_ocr(self):
        """
        this function exports the extracted information from the images of the document
        :return: a dict with { image_type : [ {field_name: extracted information} ] }
        """
        export = {}
        for image_type in self.images.keys():
            export[image_type] = []
            for image in self.images[image_type]:
                export[image_type].append(self.select_extracted_field(image.extracted_information))
        return export


class CNI(Document):  # This should be a child class from more generic class
    # TODO : traiter l'information date de validité en ajoutant un traitement du verso
    def __init__(self, recto_path=None,verso_path=None):
        super().__init__()
        if recto_path:
            self.add_image(image_path=recto_path,type='RectoCNI')
        if verso_path:
            self.add_image(image_path=verso_path, type='VersoCNI')




if __name__ == "__main__":
    import json
    cni = CNI(recto_path='data/CNI_caro3.jpg')
    cni.align_images(debug=False)
    cni.clean_images()
    cni.extract_ocr(debug=False)
    results = cni.export_ocr()
    print(json.dumps(results, indent=4, sort_keys=True))