import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
import pytesseract

from src.image_preprocessing.preprocessing import align_images
from src.image_preprocessing.remove_noise import ImageCleaning


class CNI():  # This should be a child class from more generic class
    # TODO : traiter l'information date de validité en ajoutant un traitement du verso

    def __init__(self, reference_image='data/CNI_robin.jpg', debug=False):
        # The zones are given as (x_start,y_start,x_end,y_end)
        self.zones = {
            'nom': {"value": "Nom :", "title": (448, 212, 524, 242), "field": (525, 210, 800, 270)},
            'prenom': {"value": "Prénom(s) :", "title": (448, 294, 576, 329), "field": (570, 280, 1500, 350)},
            'date_naissance': {"value": "Né(e) le :", "title": (745, 375, 850, 410), "field": (860, 370, 1060, 430)},
            'sexe': {'value': 'Sexe', 'title': (447, 381, 527, 415)},
            'taille': {'value': 'Taille :', 'title': (451, 466, 537, 492)},
            'signature': {'value': 'Signature', 'title': (449, 503, 570, 537)},
            'du_titulaire': {'value': 'du titulaire :', 'title': (449, 538, 611, 567)},
            'nationalite_francaise': {'value': 'Nationalité Française', 'title': (920, 143, 1179, 180)},
            "carte_nationale": {'value': 'CARTE NATIONALE', 'title': (130, 160, 376, 192)}
        }
        self.reference_image = cv2.imread(str(Path(reference_image)))
        self.im_clean = ImageCleaning()
        self.debug = debug

    def load_image(self, image_path, debug=False):
        """
        This function load and align the image to process with the reference image
        :param image_path: the path to the image to process
        :param debug: if True, the result of the alignment process will be displayed
        """
        image = cv2.imread(str(image_path))
        self.image_to_process = align_images(image, self.reference_image, debug=debug)

        if debug:
            aligned = imutils.resize(self.image_to_process, width=800)
            template = imutils.resize(self.reference_image, width=800)
            stacked = np.hstack([aligned, template])
            overlay = template.copy()
            output = aligned.copy()
            cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
            # show the two output image alignment visualizations
            cv2.imshow("Image Alignment Stacked", stacked)
            cv2.imshow("Image Alignment Overlay", output)
            cv2.waitKey(0)

    def tune_preprocessing(self):
        """
        this function should autotune the parameters for the denoising of the image to optimize the output of the known fields
        Maybe it should belong to a proper image cleaning class
        :return:
        """
        pass

    def extract_ocr(self):
        cleaned_image = self.im_clean.remove_noise_and_smooth(self.image_to_process)
        results = {}
        for zone in self.zones.keys():
            if 'field' in self.zones[zone].keys():
                y_min, x_min, y_max, x_max = self.zones[zone]['field']
                crop = cleaned_image[x_min:x_max, y_min:y_max]
                if self.debug:
                    cv2.imshow("ORC", crop)
                    cv2.waitKey(0)
                results[zone] = pytesseract.image_to_string(crop, lang='fra',config='--psm 7 --oem 3')
                # for parameters of tesseract refer to http://manpages.ubuntu.com/manpages/bionic/man1/tesseract.1.html
        return results


if __name__ == "__main__":
    from pathlib import Path
    cni = CNI(debug=True)
    cni.load_image(image_path=Path('data/CNI_caro.jpg'))
    cni.tune_preprocessing()
    results = cni.extract_ocr()
    print(results)