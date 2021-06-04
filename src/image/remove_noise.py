import cv2
import logging
import numpy as np

from skopt.space import Categorical, Integer, Real


BINARY_THREHOLD = 180

size = None

class ImageCleaning():
    def __init__(self):
        self.threshold_type = 'otsu'
        self.use_bilateral_filter = True
        self.use_median_filter = False
        self.use_gaussian_filter = False
        self.bilateralFilter_params = {"d": 11, "sigmaColor": 17, "sigmaSpace": 17}
        self.kernel_size = 1

    def remove_noise_and_smooth(self, img):
        logging.info('Removing noise and smoothening image')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #gray scale

        #FILTERING
        if self.use_bilateral_filter:
            img = cv2.bilateralFilter(src=img, **self.bilateralFilter_params) #filtering with sharp edges
        elif self.use_median_filter:
            img = cv2.medianBlur(img,5)
        elif self.use_gaussian_filter:
            img = cv2.GaussianBlur(img, (5, 5), 0)

        #THRESHOLDING
        if self.threshold_type == 'otsu':
            img = cv2.threshold(src=img, thresh=0, maxval=255, type= cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        elif self.threshold_type == 'adaptative_mean':
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        elif self.threshold_type == 'adaptative_gaussian':
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        #
        # kernel = np.ones((self.kernel_size,self.kernel_size), np.uint8)
        # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        #img = cv2.dilate(img,kernel,iterations = 1) #cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return img

    def tune_preprocessing(self):
        """
        this function should autotune the parameters for the denoising of the image to optimize the output of the known fields
        Maybe it should belong to a proper image cleaning class
        :return:
        """
        pass

    def get_dimensions(self):
        dim1 = Categorical(
            name="threshold_type",
            categories=["otsu", "adaptative_mean", "adaptative_gaussian"],
        )
        dim2 = Categorical(
            name="use_bilateral_filter",
            categories = [True, False],
        )
        dim3 = Categorical(
            name="use_median_filter",
            categories = [True, False],
        )
        dim4 = Categorical(
            name="use_gaussian_filter",
            categories = [True, False],
        )

        dimensions = [dim1, dim2, dim3, dim4]
        return dimensions

    def set_parameters(self, args):
        for key, val in args.items():
            setattr(self, key, val)



if __name__ == "__main__":
    from pathlib import Path

    image = Path('data/CNI_robin.jpg')
    img = cv2.imread(str(image))
    im_clean = ImageCleaning()
    img = im_clean.remove_noise_and_smooth(img)
    cv2.imshow("cropped", img)
    cv2.waitKey(0)


    print('hello')