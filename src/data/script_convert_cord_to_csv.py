import json
import os
import pandas as pd
import numpy as np

import cv2


def convert_dim(quad):
    points = np.array([
        [quad['x1'], quad['y1']],
        [quad['x2'], quad['y2']],
        [quad['x3'], quad['y3']],
        [quad['x4'], quad['y4']]
    ])
    rect = cv2.boundingRect(points)
    return rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]


dir = 'data/CORD/dev/json'
list_files = os.listdir(dir)
data_df = []
for file_name in list_files:
    with open(os.path.join(dir, file_name), 'r') as f:
        data = json.load(f)
    original_width = data['meta']['image_size']['width']
    original_height = data['meta']['image_size']['height']
    for line in data['valid_line']:
        label = line['category']
        for word in line['words']:
            w = word['text']
            min_x, min_y, max_x, max_y = convert_dim(word['quad'])
            data_df.append(
                [w, min_x / original_width, min_y / original_height, max_x / original_width, max_y / original_height, 1,
                 file_name, label, 'N.A', original_width, original_height])

df = pd.DataFrame(data_df, columns=["word", "min_x", "min_y", "max_x", "max_y", "page_id", "document_name", "label",
                                    "completed_by.email", "original_width", "original_height"])

df.to_csv('data/CORD/dev/dev.csv', sep='\t', index=False)
