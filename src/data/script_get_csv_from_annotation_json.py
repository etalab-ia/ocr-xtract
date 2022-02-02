from pathlib import Path

from src.data.annotation_utils import LabelStudioConvertor


annotation_json = Path("data/quittances/annotation/sample1.json")
annotation_json = Path("data/bulletins/preannotation/salary_preannotation_sample_first_rotated.json")
output_path = Path("data/bulletins/annotation/sample_1_rotated.csv")


df_annotation = LabelStudioConvertor(annotation_json, output_path, False).transform()
df_annotation.to_csv(output_path, index=False, sep= "\t")