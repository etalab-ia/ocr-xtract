import os
import random
import sys
import yaml

import pandas as pd
from jenkspy import JenksNaturalBreaks
from tqdm import tqdm

from src.models_training.utils_prepare import get_optimal_nb_classes

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["prepare"]

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py data-file data_folder_output\n")
        sys.exit(1)

    # Train data set split ratio
    split = params["train_split"]
    random.seed(params["seed"])
    debug = params["debug"]

    input = sys.argv[1]
    output = sys.argv[2]
    output_train = os.path.join(output, "train.csv")
    output_test = os.path.join(output, "test.csv")

    os.makedirs(os.path.join(output), exist_ok=True)

    # creates unique hash that combines document name and page number
    df = pd.read_csv(input, sep='\t')
    df['doc_pages'] = df.apply(lambda x: str(x['document_name']) + str(x['page_id']), axis=1)

    # estimate lines in document
    all_estimated_lines = []
    for page in tqdm(df['doc_pages'].unique()):
        df_page = df[df['doc_pages'] == page].copy()
        y = df_page['max_y'] * 100

        nb_class = get_optimal_nb_classes(y)

        jnb = JenksNaturalBreaks(nb_class=nb_class)
        jnb.fit(y)

        estimated_lines = jnb.predict(y)
        all_estimated_lines.extend(estimated_lines)

    df['estimated_line_number'] = all_estimated_lines

    # shuffle and split
    list_pages = df['doc_pages'].unique()
    random.shuffle(list_pages)
    nb_train = int(len(list_pages) * split)
    if debug:
        # takes very few documents
        list_pages_train = list_pages[:10]
        list_pages_test = list_pages[10:15]
    else:
        list_pages_train = list_pages[:nb_train]
        list_pages_test = list_pages[nb_train:]
    df_train = df[df['doc_pages'].isin(list_pages_train)]
    df_test = df[df['doc_pages'].isin(list_pages_test)]

    # saves
    df_train.to_csv(output_train, sep='\t', index=False)
    df_test.to_csv(output_test, sep='\t', index=False)