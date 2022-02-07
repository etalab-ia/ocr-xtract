import os
import random
import sys
import yaml
from tqdm import tqdm
from shutil import rmtree, copyfile

import pandas as pd


def generate_annotations(list_images, df):
    words = []
    boxes = []
    labels = []
    for image in tqdm(list_images):
        data = df[df['document_name'].str.contains(image)]
        if len(data) == 0:
            print('No data for ' + image)
        width, height = data["original_width"].unique()[0], data["original_height"].unique()[0]
        # loop over OCR annotations
        words.append(data['word'].to_list())
        boxes.append([[int(1000 * b) for b in d] for d in data[['min_x', 'min_y', 'max_x', 'max_y']].to_numpy().tolist()])   # important: each bounding box should be in (bottom left, upper right) format
        labels.append(data['label'].to_list())

    return words, boxes, labels


if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["prepare"]

    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py data-file image-data-folder data_folder_output\n")
        sys.exit(1)

    # Train data set split ratio
    split = params["train_split"]
    random.seed(params["seed"])
    debug = params["debug"]

    input_csv = sys.argv[1]
    input_image_folder = sys.argv[2]
    output = sys.argv[3]
    output_train = os.path.join(output, "train.pkl")
    output_test = os.path.join(output, "test.pkl")

    os.makedirs(os.path.join(output), exist_ok=True)

    # creates unique hash that combines document name and page number
    df = pd.read_csv(input_csv, sep='\t')
    df['doc_pages'] = df.apply(lambda x: str(x['document_name']) + str(x['page_id']), axis=1)
    df['word'] = df['word'].fillna(' ')

    # read images folder
    list_images = os.listdir(input_image_folder)

    # shuffle and split
    random.shuffle(list_images)
    nb_train = int(len(list_images) * split)
    if debug:
        # takes very few documents
        list_pages_train = list_images[:10]
        list_pages_test = list_images[10:15]
    else:
        list_pages_train = list_images[:nb_train]
        list_pages_test = list_images[nb_train:]

    # copy images in folder
    rmtree(os.path.join(output, 'images', 'train'), ignore_errors=True)
    os.makedirs(os.path.join(output, 'images', 'train'))
    for file in list_pages_train:
        copyfile(os.path.join(input_image_folder, file), os.path.join(output, 'images', 'train', file))

    rmtree(os.path.join(output, 'images', 'test'), ignore_errors=True)
    os.makedirs(os.path.join(output, 'images', 'test'))
    for file in list_pages_test:
        copyfile(os.path.join(input_image_folder, file), os.path.join(output, 'images', 'test', file))

    words_train, boxes_train, labels_train = generate_annotations(list_pages_train, df)
    words_test, boxes_test, labels_test = generate_annotations(list_pages_test, df)

    """# Saving"""

    import pickle

    with open(output_train, 'wb') as t:
        pickle.dump([words_train, labels_train, boxes_train], t)
    with open(output_test, 'wb') as t:
        pickle.dump([words_test, labels_test, boxes_test], t)