import difflib
import bs4 as bs

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

from src.util.debug import show_image_with_boxes

from src.image_preprocessing.preprocessing import align_images
from src.image_preprocessing.remove_noise import ImageCleaning


def parse_hocr(search_terms=None, hocr_file=None, regex=None):
    """Parse the hocr file and find a reasonable bounding box for each of the strings
    in search_terms.  Return a dictionary with values as the bounding box to be used for
    extracting the appropriate text.

    inputs:
        search_terms = Tuple, A tuple of search terms to look for in the HOCR file.

    outputs:
        box_dict = Dictionary, A dictionary whose keys are the elements of search_terms and values
        are the bounding boxes where those terms are located in the document.
    """
    # Make sure the search terms provided are a tuple.
    if not isinstance(search_terms,tuple):
        raise ValueError('The search_terms parameter must be a tuple')

    # Make sure we got a HOCR file handle when called.
    if not hocr_file:
        raise ValueError('The parser must be provided with an HOCR file handle.')

    # Open the hocr file, read it into BeautifulSoup and extract all the ocr words.
    hocr = open(hocr_file,'r').read()
    soup = bs.BeautifulSoup(hocr,'html.parser')
    words = soup.find_all('span',class_='ocrx_word')

    result = dict()

    # Loop through all the words and look for our search terms.
    for word in words:

        w = word.get_text().lower()

        for s in search_terms:

            # If the word is in our search terms, find the bounding box
            if len(w) > 1 and difflib.SequenceMatcher(None, s, w).ratio() > .5:
                bbox = word['title'].split(';')
                bbox = bbox[0].split(' ')
                bbox = tuple([int(x) for x in bbox[1:]])

                # Update the result dictionary or raise an error if the search term is in there twice.
                if s not in result.keys():
                    result.update({s:bbox})

            else:
                pass

    return result

if __name__ == "__main__":
    from pathlib import Path
    import cv2
    image = Path('data/CNI_robin_clean.jpg')
    hocr = pytesseract.image_to_pdf_or_hocr(str(image), lang='fra',extension='hocr')
    hocr_file = image.with_suffix('.xml')
    with open(hocr_file, 'wb') as f:
        f.write(hocr)
    parse_hocr(search_terms=('Prénom',), hocr_file=hocr_file)

    img = cv2.imread(str(image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.GaussianBlur(img, (5,5), 0)
    img = cv2.medianBlur(img, 3)
    # img = cv2.bilateralFilter(img, 9, 75, 75)
    # cv2.imshow("cropped", img)
    # cv2.waitKey(0)
    #_, img = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    #img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imshow("cropped", img)
    cv2.waitKey(0)
    crop = img[200:260,541:700]
    cv2.imshow("cropped", crop)
    cv2.waitKey(0)

    print(pytesseract.image_to_string(crop))

    print('hello')