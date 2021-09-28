from pathlib import Path

from src.data.annotation_utils import LabelStudioConvertor

annotation_json = Path("data/salary/annotated_train.json")
output_path = Path("data/salary/annotated_train.csv")

LabelStudioConvertor(annotation_json, output_path).transform()