OCR_XTRACT
====
![LOGO](.github/reading_snake.jpg)

This project is conducted by the [Lab IA](https://www.etalab.gouv.fr/datasciences-et-intelligence-artificielle) at [Etalab](https://www.etalab.gouv.fr/).  
The aim of the Lab IA is to help the french administration to modernize its services by the use of modern AI techniques.  
Other Lab IA projects can be found at the [main GitHub repo](https://github.com/etalab-ia/). 
#### -- Project Status: [Active]

## OCR Xtract
OCR-Xtract is a tool to extract information from administrative documents. It is meant to ease the work of state agents willing to validate administrative dossiers. OCR_Xtract will consist in :
- A front-end for uploading files (not included in this repo)
- An API to access the extracting logic
- The code to extract the information from the scanned images. 

### Methods Used
* OCR
* Image Processing
### Technologies 
* Python

## Project Description 
For now, only a POC is avaible for extracting information for French DNI 

## Getting Started for development
* Fork this repo 
* Install requirements : `pip install -r requirements.txt`

Since we use [doctr](https://mindee.github.io/doctr/), you will need extra dependencies if you are not running Linux.
### For MacOS users
You can install them as follows:
```shell
brew install cairo pango gdk-pixbuf libffi
```
###For Windows users
Those dependencies are included in GTK. You can find the latest installer over [here](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases).

We also use `pdf2image`. For installing the requirements  are to install `poppler`

### Windows
Windows users will have to build or download poppler for Windows. I recommend [@oschwartz10612 version](https://github.com/oschwartz10612/poppler-windows/releases/) which is the most up-to-date. You will then have to add the `bin/` folder to [PATH](https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/) or use `poppler_path = r"C:\path\to\poppler-xx\bin" as an argument` in `convert_from_path`.

### Mac
Mac users will have to install [poppler for Mac](http://macappstore.org/poppler/).

### Linux
Most distros ship with `pdftoppm` and `pdftocairo`. If they are not installed, refer to your package manager to install `poppler-utils`

## How to extract information 
### CNI
Using the reference CNI in `/tutorials/model_CNI.png` and point to it when creating a `RectoCNI` class :
```Python
from src.image.image import RectoCNI
image = RectoCNI('data\CNI_caro2.jpg', reference_path='data/reference.png')
image.extract_information()
```

### Using app (dev)
Launch APP using
```
streamlit run app_local.py
```

You can launch the app via the Dockerfile

## How to perform the annotation
### CNI 
1. Prepare the data for annotation: 
   Launch script `align_cni_in_folder.py`. This script will align the images. The CNI are easier to detect once they have been aligned to the reference CNI.
2. Prepare the annotated data : 
   Launch script `script_prepare_CNI_annotation.py`. This script will create a csv file with the OCR data extracted from image and to be annotated with Label Studio
3. Place the images to be annotated in `/data/label_studio_files` 
4. Launch the annotation platform. This command will mount your local folder data\label-studio inside label-studio so label-studio can use it. 
   It will also mount the folder label_studio_files where you have put the images to be annotated inside label-studio
```
docker run -it -p 8080:8080 -v `pwd`/data/label-studio:/label-studio/data \
--env LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \ 
--env LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files \ 
-v `pwd`/data/label_studio_files:/label-studio/files \
heartexlabs/label-studio:latest

docker run -it -p 8080:8080 -v C:\Users\Utilisateur\PythonProjects\ocr-xtract\data\label-studio:/label-studio/data --env LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true --env LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files -v C:\Users\Utilisateur\PythonProjects\ocr-xtract\data\label_studio_files:/label-studio/files heartexlabs/label-studio:latest
```

5. Create an annotation project :
   - import the json file generated in step 2.
   - select object detection with Bounding Boxes in Labeling Setup
   - Input the label you want to have for the annotation.

5. Once the annotation is complete, you can export the annotation in json
6. Convert the annotation from json to csv with `script_get_csv_from_annotation_json.py`
7. Use this file to train a new model with ``train_cni_pipeline.py``

## How to convert the label studio annotated file into csv file 

- export label studio annotation using the Label Studio json format 

`df = LabelStudioConvertor(Path("export.json"), Path("annotated_data.csv")).transform()`

where : 
- export.json : path to the exported label studio json file
- annotated_data.csv : path where you want to save the csv file 


## Contributing Lab IA Members
* [R. Reynaud](https://github.com/rob192)
* [G. Santarsieri](https://github.com/giuliasantarsieri)
* [P. Soriano](https://github.com/psorianom)
* [K. Montalibet](https://github.com/orgs/etalab-ia/people/KimMontalibet)

## How to contribute to this project 
We love your input! We want to make contributing to this project as easy and transparent as possible : see our [contribution rules](https://github.com/etalab-ia/ocr-xtract/blob/master/.github/contributing.md)
