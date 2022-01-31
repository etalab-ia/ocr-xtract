import os
from pickle import load
import json
import yaml
import shutil
from tempfile import mkdtemp

import numpy as np
import cv2
import matplotlib.pyplot as plt

from pathlib import Path

import PIL.Image as Image

from src.models_training.utils import select_candidate

Image.MAX_IMAGE_PIXELS = None
# used to bypass PIL.Image.DecompressionBombError that prevents from opening large files

from src.image.preprocessing import convert_pdf_to_image

from src.data.doctr_utils import DoctrTransformer, AnnotationDatasetCreator
from src.postprocessing.postprocessing_cni import clean_date, clean_names

class Image():
    def __init__(self, image_path, folder):

        self.image_path = Path(image_path)
        if self.image_path.exists():
            self.original_image = self.load_image(self.image_path)
            self.aligned_image = None
            self.cleaned_image = None
        else:
            print("Image path does not exist")

        self.folder = folder
        scheme_path = os.path.join(folder, "scheme.json")
        with open(scheme_path, 'rb') as f_s:
            self.scheme = json.load(f_s)

        self.params = yaml.safe_load(open(os.path.join(folder, "params.yaml")))
        reference_path = self.params['reference_path']
        if reference_path is not None:
            self.reference_path = Path(reference_path)
            try:
                self.reference_image = self.load_image(self.reference_path)
            except:
                print("Reference path does not exists")
        else:
            self.reference_path = None

        pipe_file = os.path.join(folder, "features/pipe.pickle")
        with open(pipe_file, 'rb') as f1:
            self.pipe_feature = load(f1)

        self.extracted_information = {}
        self.doctr_transformer = DoctrTransformer()



    def load_image(self, image_path):
        if self.image_path.exists():
            if image_path.suffix == '.pdf':
                img = np.array(convert_pdf_to_image(image_path)[0])[:, :, ::-1] #convert to numpy and BGR instead of RGB
            else:
                img = cv2.imread(str(image_path))

            # reduce image size when it's too large:
            max_size = 25000000
            if img.size > max_size:
                scaling = np.sqrt(max_size / img.size)
                img = cv2.resize(img, (0,0), fx=scaling, fy=scaling, interpolation=cv2.INTER_NEAREST)
            return img
        else:
            print('image path does not exists')


    def save(self, image_path):
        pre, ext = os.path.splitext(image_path)
        if ext == '.pdf':
            #if original image is a pdf, save as a png
            image_path = pre + '.png'
        if self.aligned_image is not None:
            cv2.imwrite(image_path, self.aligned_image)
        elif self.cleaned_image is not None:
            cv2.imwrite(image_path, self.cleaned_image)
        else:
            cv2.imwrite(image_path, self.original_image)


    def align_images(self, debug=False):
        if self.reference_path is not None:
            print(f'Aligning image with the reference now ...{self.reference_path}')
            # convert both the input image and template to grayscale
            imageGray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            templateGray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)

            # rescale template so the original quality of the image is not lost
            ho, wo = imageGray.shape
            ht, wt = templateGray.shape
            scaling = min(ho/ht, wo/wt)
            templateGray = cv2.resize(templateGray,None,fx=scaling, fy=scaling, interpolation = cv2.INTER_LINEAR)
            # Initiate SIFT detector
            sift = cv2.SIFT_create()
            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(templateGray, None)
            kp2, des2 = sift.detectAndCompute(imageGray, None)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=100)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            # store all the good matches as per Lowe's ratio test.
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            try:
                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                h, w = templateGray.shape
                # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                # dst = cv2.perspectiveTransform(pts, M)
                # im2 = cv2.polylines(self.original_image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
                self.aligned_image = cv2.warpPerspective(self.original_image, M, (w,h))
            except:
                self.aligned_image = None
                print('Image cannot be aligned')

            if debug:
                matchesMask = mask.ravel().tolist()
                draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                   singlePointColor=None,
                                   matchesMask=matchesMask,  # draw only inliers
                                   flags=2)
                img3 = cv2.drawMatches(templateGray, kp1, imageGray, kp2, good, None, **draw_params)
                plt.imshow(img3, 'gray'), plt.show()

        elif type(self.original_image) is not None:
            print("No reference Image to align original image with")
            self.aligned_image = self.original_image
        else:
            print('No original Image to align')

        return self.aligned_image

    def extract_results (self, X):
        """
        get the results from labelisation
        :param X: dataframe
        :return:
        """
        extracted_information ={}
        list_info = X['label'].unique()
        for info in list_info:
            # TODO : once we'll have line identification, we need to read by lines
            list_words = X[X['label'] == info].sort_values(['min_x'])['word'].to_list()
            extracted_information[info] = {}
            extracted_information[info]['field'] = list_words
        return extracted_information

    def extract_information(self, debug=False):
        if self.aligned_image is None:
            self.align_images(debug=debug)

        temp_folder = mkdtemp()
        cv2.imwrite(os.path.join(temp_folder, 'temp.jpg'), self.aligned_image)
        doc = DoctrTransformer().transform([Path(os.path.join(temp_folder, 'temp.jpg'))])
        dataset_creator = AnnotationDatasetCreator()
        X = dataset_creator.transform(doc)
        del doc
        del dataset_creator
        shutil.rmtree(temp_folder)

        X_feats = self.pipe_feature.transform(X)
        features = self.pipe_feature.get_feature_names()

        # Debug
        X.loc[:, features] = X_feats

        model_folder = os.path.join(self.folder, "model")
        list_model = os.listdir(model_folder)

        extracted_information = {}

        for candidate_name in self.scheme.keys():
            print(candidate_name)
            training_field = self.scheme[candidate_name]['training_field']
            candidate_feature = self.scheme[candidate_name]['candidate']

            list_model_candidate = [m for m in list_model if candidate_name == m.split('-')[0]]

            extracted_information[candidate_name] = {}

            model_name = list_model_candidate[0] # TODO have eval select best model
            with open(os.path.join(model_folder, model_name), 'rb') as f:
                model_data = load(f)

            model = model_data['model']

            X_candidate, iscandidate = select_candidate(candidate_name, candidate_feature, features, X_feats)
            X.loc[iscandidate, candidate_name] = model.predict(X_candidate)


            list_words = X[X[candidate_name] == 1].sort_values(['min_x'])['word'].to_list()

            extracted_information[candidate_name]['field'] = list_words

        self.cleaned_results = self.clean_results(extracted_information)
        return self.cleaned_results

    def clean_results(self, extracted_information):
        """clean results """
        cleaned_results = extracted_information
        return cleaned_results

    def reset(self):
        self.__init__()




if __name__ == "__main__":
    from datetime import datetime

    global start
    start = datetime.now()
    print(start)
<<<<<<< HEAD
    # image = Image('./data/salary/test/3fc1665f-af1b-4ced-9194-19610f1debe4.jpg', 'data_dvc/salary')
    image = Image('./data/CNI_76e4a9a1-2eda-4cd9-9f37-28f773336bb1.png', 'data_dvc/cni_recto')
=======
    # image = Image('./tutorials/model_CNI.png', 'data_dvc/cni_recto')
    # image = Image('./data/salary/test/3fc1665f-af1b-4ced-9194-19610f1debe4.jpg', 'data_dvc/salary')
    image = Image('./data/quittances/sample1/b4587d5d-0165-4fbd-9c7e-cfdeacd6575a.jpg', './data_dvc/rent_receipts')
>>>>>>> straightened_annotated_images
    image.align_images()
    response = image.extract_information()
    print(response)