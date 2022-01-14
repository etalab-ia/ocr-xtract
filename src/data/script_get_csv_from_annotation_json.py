from pathlib import Path

from src.data.annotation_utils import LabelStudioConvertor

annotation_json = Path("data/quittances/annotation/sample1_2.json")
output_path = Path("data/quittances/annotation/sample1_2.csv")

df_annotation = LabelStudioConvertor(annotation_json, output_path).transform()
df_annotation.to_csv(output_path, index=False, sep= "\t")