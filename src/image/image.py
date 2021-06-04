import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract
from pathlib import Path
from collections import Counter

from skopt import dump, gp_minimize
from skopt.utils import use_named_args
from skopt.plots import plot_objective

from src.image.remove_noise import ImageCleaning
from src.image.utils_optimizer import LoggingCallback


class Image():
    def __init__(self, image_path, reference_path=None):
        self.image_cleaner = ImageCleaning()
        self.image_path = Path(image_path)
        self.reference_path = Path(reference_path)
        if self.image_path.exists():
            self.original_image = self.load_image(self.image_path)
            self.aligned_image = None
            self.cleaned_image = None
        else:
            print("Image path does not exist")
        if self.reference_path.exists():
            self.reference_image = self.load_image(self.reference_path)
        else:
            print("Reference path does not exists")
        self.zones = {}
        self.extracted_information = {}

    def load_image(self, image_path):
        return cv2.imread(str(image_path))

    def save(self):
        pass

    def align_images(self, debug=False):
        if type(self.original_image) is not None:
            print('Aligning image with the reference now ...')
            # convert both the input image and template to grayscale
            imageGray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            templateGray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)

            # Initiate SIFT detector
            sift = cv2.SIFT_create()
            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(templateGray, None)
            kp2, des2 = sift.detectAndCompute(imageGray, None)

            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            # store all the good matches as per Lowe's ratio test.
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w, d = self.reference_image.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            # im2 = cv2.polylines(image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            self.aligned_image = cv2.warpPerspective(self.original_image, M, (w, h))

            if debug:
                draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                   singlePointColor=None,
                                   matchesMask=matchesMask,  # draw only inliers
                                   flags=2)
                img3 = cv2.drawMatches(templateGray, kp1, imageGray, kp2, good, None, **draw_params)
                plt.imshow(img3, 'gray'), plt.show()


        else:
            print('No original Image to align')

        return self.aligned_image

    def clean(self, debug=False):
        """
        Clean self.image with the image_cleaner
        :return: self.image cleaned
        """
        if self.aligned_image is None:
            self.align_images()

        self.cleaned_image = self.image_cleaner.remove_noise_and_smooth(self.aligned_image)

        # TODO : implement a check that the image was properly cleaned. If not, trigger self.tune_preprocessing
        if debug:
            cv2.imshow('Cleaned image', self.cleaned_image)
            cv2.waitKey(0)
        return self.cleaned_image

    def ocr_field(self,zone, field, debug=False):
        """
        Ocr field where field is a tuple from zone in the format (x_start,y_start,x_end,y_end)
        :param zone: name of the zone to extract (string)
        :param field: tuple from zone in the format (x_start,y_start,x_end,y_end)
        :param debug: if True, a window will appear with the field OCRed
        :return:
        """
        y_min, x_min, y_max, x_max = self.zones[zone][field]
        crop = self.cleaned_image[x_min:x_max, y_min:y_max]
        if debug:
            cv2.imshow("ORC", crop)
            cv2.waitKey(0)
        # for parameters of tesseract refer to http://manpages.ubuntu.com/manpages/bionic/man1/tesseract.1.html
        return pytesseract.image_to_string(crop, lang='fra', config='--psm 7 --oem 3')

    def extract_ocr(self, ocr_unknown_fields_only=False, debug=False):
        if self.cleaned_image is None:
            self.clean()

        for zone in self.zones.keys():
            self.extracted_information[zone] = {}
            if 'title' in self.zones[zone].keys():
                if ocr_unknown_fields_only == True:
                    pass
                else:
                    self.extracted_information[zone]['title'] = self.ocr_field(zone,'title', debug)
            if 'field' in self.zones[zone].keys():
                self.extracted_information[zone]['field'] = self.ocr_field(zone,'field', debug)
        return self.extracted_information

    def quality_ocr(self):
        def shared_chars(s1, s2):
            return sum((Counter(s1) & Counter(s2)).values())

        self.extract_ocr()
        sum_quality = 0
        nb_zones = 0

        for zone in self.extracted_information.keys():
            true_positives = shared_chars(self.extracted_information[zone]['title'],self.zones[zone]['value'])
            false_positives = len(self.extracted_information[zone]['title']) - true_positives
            false_negatives = len(self.zones[zone]['value']) - true_positives
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f1 = 2 * (precision * recall) / (precision + recall)
            sum_quality += f1
            nb_zones += 1
        return sum_quality / nb_zones

    def reset(self):
        self.__init__()


    def tune_preprocessing(self, debug=False):
        image = self
        dimensions = self.image_cleaner.get_dimensions()
        @use_named_args(dimensions=dimensions)
        def read_known_field(**kwargs):
            image.image_cleaner.set_parameters(kwargs)
            image.clean()
            opt = 1 - image.quality_ocr()
            return opt

        n_calls = 10

        res = gp_minimize(
            read_known_field,
            dimensions,
            n_calls=n_calls,
            callback=LoggingCallback(n_calls),
            n_jobs=-1,
        )

        if debug == True:
            plot_objective(res)
            plt.show()

        best_score = read_known_field(res.x) # Allow cleaning of the image with best config
        return best_score



class RectoCNI(Image):
    def __init__(self, image_path=None, reference_path='data/CNI_robin.jpg'):
        super().__init__(image_path, reference_path)
        self.zones = {
            'nom': {"value": "Nom :", "title": (448, 212, 524, 242), "field": (525, 210, 800, 270)},
            'prenom': {"value": "Prénom(s) :", "title": (448, 294, 576, 329), "field": (570, 280, 1500, 350)},
            'date_de_naissance': {"value": "Né(e) le :", "title": (745, 375, 850, 410),
                                  "field": (860, 370, 1060, 430)},
            'sexe': {'value': 'Sexe', 'title': (447, 381, 527, 415)},
            'taille': {'value': 'Taille :', 'title': (451, 466, 537, 492)},
            'signature': {'value': 'Signature', 'title': (449, 503, 570, 537)},
            'du_titulaire': {'value': 'du titulaire :', 'title': (449, 538, 611, 567)},
            'nationalite_francaise': {'value': 'Nationalité Française', 'title': (920, 143, 1179, 180)},
            "carte_nationale": {'value': 'CARTE NATIONALE', 'title': (130, 160, 376, 192)}
        }
        self.image_cleaner.tune_preprocessing()

class VersoCNI(Image):
    def __init__(self, image_path=None, reference_path='data/CNI_robin_verso.jpg'):
        super.__init__(image_path, reference_path)
        self.zones = {
            'date_expiration': {"value": "Carte valable jusqu'au :", "title": (), "field": ()},
            'adresse': {"value": "Adresse :", "title": ()}
        }
        self.image_cleaner.tune_preprocessing()


if __name__ == "__main__":
    image = RectoCNI('data/CNI_caro3.jpg')
    image.align_images()
    image.clean()
    image.extract_ocr()
    image.quality_ocr()
    best_score = image.tune_preprocessing()
    print(best_score)