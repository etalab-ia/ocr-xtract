import pytesseract
import cv2


def show_image_with_boxes(originalImg, boxes):
    h, w = originalImg.shape
    for box in boxes:
        cv2.rectangle(originalImg, (int(box[1]), h - int(box[2])), (int(box[3]), h - int(box[4])), (0,0,255), 1)
    cv2.imshow('', originalImg)
