import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
from pdf2image import convert_from_path


def convert_pdf_to_image(pdf):
    """
    This function uses the
    :param pdf:
    :return:
    """
    pages = convert_from_path('pdf_file', 500)
    return pages


def align_images(image, template, debug=False):
    # convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

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
    h, w, d = template.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    # im2 = cv2.polylines(image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    aligned = cv2.warpPerspective(image, M, (w, h))

    if debug:
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)
        img3 = cv2.drawMatches(templateGray, kp1, imageGray, kp2, good, None, **draw_params)
        plt.imshow(img3, 'gray'), plt.show()

    return aligned


if __name__ == "__main__":
    from pathlib import Path

    image = image = cv2.imread(str(Path("data/CNI_caro3.jpg")))
    template = cv2.imread(str(Path("data/CNI_robin.jpg")))

    aligned = align_images(image, template, debug=True)

    # resize both the aligned and template images so we can easily
    # visualize them on our screen
    aligned = imutils.resize(aligned, width=800)
    template = imutils.resize(template, width=800)

    plt.imshow(aligned)
    plt.show()
    # our first output visualization of the image alignment will be a
    # side-by-side comparison of the output aligned image and the
    # template
    stacked = np.hstack([aligned, template])

    # our second image alignment visualization will be *overlaying* the
    # aligned image on the template, that way we can obtain an idea of
    # how good our image alignment is
    overlay = template.copy()
    output = aligned.copy()
    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    # show the two output image alignment visualizations
    cv2.imshow("Image Alignment Stacked", stacked)
    cv2.imshow("Image Alignment Overlay", output)
    cv2.waitKey(0)
