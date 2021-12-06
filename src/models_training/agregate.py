import os
import random
import sys
import yaml
import json

import pandas as pd

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython agregate.py datasets-list data_folder_output\n")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = os.path.join(sys.argv[2], 'data_set.csv')

    with open(input_file, 'r') as f:
        datasets = json.load(f)['datasets']

    df = pd.DataFrame([])
    for dataset in datasets:
        df1 = pd.read_csv(dataset, sep='\t')
        df = pd.concat([df, df1], axis=0, ignore_index=True)

    df['doc_pages'] = df.apply(lambda x: str(x['document_name']) + str(x['page_id']), axis=1)
    df = df.groupby(['doc_pages'])['word'].apply(list).reset_index()

    df.to_csv(output_file, sep='\t')