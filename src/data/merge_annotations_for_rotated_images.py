import random
import json
import pandas as pd
import os
from pathlib import Path

# path to the csv with the original annotation
path_df_annotations = Path("data_dvc/salary/full_data_set.csv", sep = "\t")
# path to the csv with the doctr boxes generated from the rotated images, therefore the labels are only "0"
path_df_rotated = Path("data/bulletins/annotation/sample_1_rotated.csv")
# Output path of the merged dataset
output_path = "data/bulletins/annotation/full_data_set_rotated_with_labels.csv"

def main():

    df = pd.read_csv(path_df_annotations, sep = "\t")
    dfr = pd.read_csv(path_df_rotated, sep = "\t")

    # drop label columns as it contains only "O"
    dfr = dfr.drop(["label"], axis = 1)
    #  keep in df only the documents present in the rotated df
    list_doc_dfr = list(set(dfr.document_name))
    df = df[df["document_name"].isin(list_doc_dfr)]

    ## keep only words with label in df
    dff = df[df["label"]!="O"]
    dff = dff[['word', 'page_id', 'document_name', 'label']]

    # drop duplicated rows
    dff_nodup = dff.drop_duplicates(subset =['word', 'page_id', 'document_name', "label"])
    df_merged = pd.merge(dfr, dff_nodup, on = ['word', 'page_id', 'document_name'], how = "left")
    nb_row_merged = len(pd.merge(dfr, dff_nodup, on = ['word', 'page_id', 'document_name']))
    print("{} rows out of {} were merged, which makes {}%".format(nb_row_merged, len(dff_nodup), nb_row_merged/len(dff_nodup)))
    df_merged.to_csv(output_path, index=False, sep="\t")

if __name__ == '__main__':
    main()
