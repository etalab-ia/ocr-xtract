OCR_XTRACT
====

This project is conducted by the [Lab IA](https://www.etalab.gouv.fr/datasciences-et-intelligence-artificielle) at [Etalab](https://www.etalab.gouv.fr/).  
The Lab IA helps french administrations to modernize their services by the use of modern AI techniques.  
Other Lab IA projects can be found on our [GitHub organization](https://github.com/etalab-ia/). 

#### -- Project Status: [Active]

## OCR Xtract
OCR-Xtract is a tool that performs OCR and information extraction from documents. It is meant to speed up the work of state agents dealing with documents whose formats are not directly numerically exploitable. OCR_Xtract will consist in :
- A front-end for uploading files 
- An API to access the trained model for the Key Information Extraction
- The code to extract the information from the scanned images. 

### Methods Used
* OCR
* Image Processing


### Technologies 
* Python 3.7

## Project Description 
For now, only a POC is available for extracting information for French DNI and for the french payslips. 

## Getting Started for development
* Fork this repo
* Update pip : `pip install --upgrade pip`
* Install requirements : `pip install -r requirements.txt`

### Install Doctr
Since we use [doctr](https://mindee.github.io/doctr/), you will need extra dependencies.
We also use `pdf2image`, so you will have to install poppler. 

#### For MacOS users
You can install them as follows:
```shell
brew install cairo pango gdk-pixbuf libffi
```
Mac users will have to install [poppler for Mac](http://macappstore.org/poppler/).
Install poppler with the command 
```shell
brew install poppler
``` 
If this one does not work, an alternative is to use conda : 
```shell
conda install -c conda-forge poppler
``` 


#### For Windows users
Those dependencies are included in GTK. You can find the latest installer over [here](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases).
Windows users will have to build or download poppler for Windows. I recommend [@oschwartz10612 version](https://github.com/oschwartz10612/poppler-windows/releases/) which is the most up-to-date. You will then have to add the `bin/` folder to [PATH](https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/) or use `poppler_path = r"C:\path\to\poppler-xx\bin" as an argument` in `convert_from_path`.

#### Linux
If you experience trouble with Weasyprint and pango, install with this
```apt install python3-pip python3-cffi python3-brotli libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0 libgl1-mesa-glx```
Most distros ship with `pdftoppm` and `pdftocairo`. If they are not installed, refer to your package manager to install `poppler-utils`

### Install DVC (Optional)
To have access to the datasets on which the models are trained, install dvc and connect to our s3 bucket.

```shell
dvc remote add -d minio s3://labia/PATH/TO/STORE -f --local
dvc remote modify minio endpointurl https://our.endpoint.fr --local
dvc remote modify minio access_key_id ourkey --local
dvc remote modify minio secret_access_key ourpassword --local
dvc pull
```


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
--env-file .env
-v `pwd`/data/label_studio_files:/label-studio/files \
heartexlabs/label-studio:latest

docker run -it -p 8080:8080 -v C:\Users\Utilisateur\PythonProjects\ocr-xtract\data\label-studio:/label-studio/data --env-file .env -v C:\Users\Utilisateur\PythonProjects\ocr-xtract\data\label_studio_files:/label-studio/files heartexlabs/label-studio:latest
```

5. Create an annotation project :
   - name your project
   - import the json file generated in step 2.
   - select object detection with Bounding Boxes in Labeling Setup
   - define the labels corresponding to the categories you want to extract 

5. Once the annotation is complete, you can export the annotation in the json format
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
