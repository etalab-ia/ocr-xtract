import cv2
import logging
import numpy as np

BINARY_THREHOLD = 180

size = None

class ImageCleaning():
    def __init__(self):
        self.bilateralFilter_params = {"d": 11, "sigmaColor": 17, "sigmaSpace": 17}
        self.threshold_params = {"type": cv2.THRESH_BINARY + cv2.THRESH_OTSU}
        self.kernel_size = 1

    def remove_noise_and_smooth(self, img):
        logging.info('Removing noise and smoothening image')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #gray scale
        img = cv2.bilateralFilter(src=img,**self.bilateralFilter_params) #filtering with sharp edges
        img = cv2.threshold(src=img, thresh=0, maxval=255, **self.threshold_params)[1]
        #Thresholding
        kernel = np.ones((self.kernel_size,self.kernel_size), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        #img = cv2.dilate(img,kernel,iterations = 1) #cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return img

if __name__ == "__main__":
    from pathlib import Path

    image = Path('data/CNI_robin.jpg')
    img = cv2.imread(str(image))
    im_clean = ImageCleaning()
    img = im_clean.remove_noise_and_smooth(img)
    cv2.imshow("cropped", img)
    cv2.waitKey(0)


    print('hello')