# Preannotation with Doctr / Annotation with Label Studio 

## How to create preannotated file in the Label Studio Format with Doctr bounding boxes

Exemple for CNI
1. Prepare the data for annotation
The CNI are easier to detect once they have been aligned to the reference CNI. First align and save your CNI with the script `align_cni_in_folder.py`

2. Prepare the annotated data with script_prepare_CNI_annotation.py

3. Place the images to be annotated in /data/label_studio_files

3. Run docker. This command will mount your local folder data\label-studio inside label-studio so label-studio can use it. 
   It will also mount the folder label_studio_files where you have put the images to be annotated inside label-studio
```
docker run -it -p 8080:8080 -v `pwd`/data/label-studio:/label-studio/data \
--env LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \ 
--env LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files \ 
-v `pwd`/data/label_studio_files:/label-studio/files \
heartexlabs/label-studio:latest

docker run -it -p 8080:8080 -v C:\Users\Utilisateur\PythonProjects\ocr-xtract\data\label-studio:/label-studio/data --env LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true --env LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files -v C:\Users\Utilisateur\PythonProjects\ocr-xtract\data\label_studio_files:/label-studio/files heartexlabs/label-studio:latest
```

4. Create an annotation project :
   - import the json file generated in step 2.
   - select object detection with Bounding Boxes in Labeling Setup
   - Input the label you want to have for the annotation.

5. 
## How to convert the label studio annotated file into csv file 

- export label studio annotation using the Label Studio json format 

`df = LabelStudioConvertor(Path("export.json"), Path("annotated_data.csv")).transform()`

where : 
- export.json : path to the exported label studio json file
- annotated_data.csv : path where you want to save the csv file 

