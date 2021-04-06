import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
import pytesseract

from src.image_preprocessing.preprocessing import align_images
from src.image_preprocessing.remove_noise import remove_noise_and_smooth


class CNI(): #This should be a child class from more generic class
    def __init__(self, params='basic'):
        if params == 'basic':
            self.zones = {'nom': {"title": (434, 217, 541, 268), "field": (525, 210, 800, 270)},
                         'prénom': {"title": (454, 302, 573, 325), "field": (570, 280, 1500, 350)}
                        }
            self.reference_image = cv2.imread(str(Path('data/CNI_robin.jpg')))
            self.preprocessing_parameters = ()

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
        :return:
        """
        pass

    def extract_ocr (self):
        cleaned_image = remove_noise_and_smooth(self.image_to_process)
        results = {}
        for zone in self.zones.keys():
            y_min, x_min, y_max, x_max = self.zones[zone]['field']
            crop = cleaned_image[x_min:x_max, y_min:y_max]
            results[zone] = pytesseract.image_to_string(crop, lang='fra')
        return results


if __name__ == "__main__":
    from pathlib import Path
    cni = CNI(params='basic')
    cni.load_image(image_path=Path('data/CNI_caro.jpg'))
    cni.tune_preprocessing()
    results = cni.extract_ocr()
