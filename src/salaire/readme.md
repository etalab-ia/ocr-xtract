# Preannotation with Doctr / Annotation with Label Studio 

## How to create preannotated file in the Label Studio Format with Doctr bounding boxes

Exemple for CNI
1. Prepare the data for annotation
The CNI are easier to detect once they have been aligned to the reference CNI. First align and save your CNI with the script `align_cni_in_folder.py`

2. Prepare the annotated data with script_prepare_CNI_annotation.py

3. Run docker 
```
docker run -it -p 8080:8080 -v `pwd`/data/annotation:/label-studio/data heartexlabs/label-studio:latest
```

4. 

## How to convert the label studio annotated file into csv file 

- export label studio annotation using the Label Studio json format 

`df = LabelStudioConvertor(Path("export.json"), Path("annotated_data.csv")).transform()`

where : 
- export.json : path to the exported label studio json file
- annotated_data.csv : path where you want to save the csv file 

