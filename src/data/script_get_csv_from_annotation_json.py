from pathlib import Path

from src.salaire.annotation_utils import LabelStudioConvertor

annotation_json = Path("data/CNI_recto_aligned_linux/train_annotated.json")
output_path = Path("data/CNI_recto_aligned_linux/train_annotated.csv")

LabelStudioConvertor(annotation_json, output_path).transform()