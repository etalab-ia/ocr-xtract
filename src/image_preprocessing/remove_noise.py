import cv2
import logging
import numpy as np

BINARY_THREHOLD = 180

size = None


def remove_noise_and_smooth(img):
    logging.info('Removing noise and smoothening image')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bilateralFilter(img,11,17,17)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((1,1), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #img = cv2.dilate(img,kernel,iterations = 1) #cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

if __name__ == "__main__":
    from pathlib import Path

    image = Path('data/CNI_robin.jpg')
    img = cv2.imread(str(image))
    img = remove_noise_and_smooth(img)
    cv2.imshow("cropped", img)
    cv2.waitKey(0)


    print('hello')