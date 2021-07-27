# Preannotation with Doctr / Annotation with Label Studio 

## How to create preannotated file in the Label Studio Format with Doctr bounding boxes

Exemple for CNI
1. Prepare the data for annotation
The CNI are easier to detect once they have been aligned to the reference CNI. First align and save your CNI with the script `align_cni_in_folder.py`

- `from annotation_utils import DoctrTransformer, AnnotationJsonCreator`
- `list_img_path = [Path(x) for x in os.listdir(img_folder_path)]`
- `doctr_transformer = DoctrTransformer()`
- `list_doctr_docs = doctr_transformer._get_doctr_docs(list_img_path)`
- `annotations = AnnotationJsonCreator(list_img_path, output_path).transform(list_doctr_docs)`

where : 
img_folder_path : forlder where your images are in jpg, png, jpeg format
output_path: path/name_preannotation.json


## How to convert the label studio annotated file into csv file 

- export label studio annotation using the Label Studio json format 

`df = LabelStudioConvertor(Path("export.json"), Path("annotated_data.csv")).transform()`

where : 
- export.json : path to the exported label studio json file
- annotated_data.csv : path where you want to save the csv file 

