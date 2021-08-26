"""
This script is used to verify if the performances of the model for detecting the content of the CNI are improved when the CNI are already aligned
"""
from pathlib import Path

from src.image.image import RectoCNI
import os

folder = Path("./data/batch_1_Kim/CNI_recto_train/")
save_folder = Path("./data/batch_1_Kim/CNI_recto_train_aligned/")

os.makedirs(save_folder, exist_ok=True)

list_files = os.listdir(folder)
list_path_to_treat = [os.path.join(folder, path) for path in list_files]
list_files_output = os.listdir(save_folder)
list_path_output = [os.path.join(folder, path) for path in list_files_output]

for path, file in zip(list_path_to_treat, list_files):
    if path[-3:] in ['jpg','pdf','png','peg'] and path not in list_path_output:
        print(f'Treating file {path}')
        image = RectoCNI(path)
        image.align_images()
        image.save(os.path.join(save_folder, file))
    else:
        print(f'Skipping file {path}')

