from pathlib import Path

from src.salaire.annotation_utils import LabelStudioConvertor

annotation_json = Path("data/salary/annotated_test.json")
output_path = Path("data/salary/annotated_test.csv")

LabelStudioConvertor(annotation_json, output_path).transform()