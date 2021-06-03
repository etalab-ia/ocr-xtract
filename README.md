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

## How to extract information 
### CNI
Place a reference CNI in `/data` and point to it when creating a `CNI` class :
```Python
from src.document.CNI import CNI
cni = CNI(recto_path=filename)
cni.align_images()
cni.clean_images()
cni.extract_ocr()
results = cni.export_ocr()
```

### Using app (dev)
Launch API using 
```
python api/app.py
```

Launch FRONT using
```
streamlit run front/app.py
```

## Contributing Lab IA Members
* [R. Reynaud](https://github.com/rob192)
* [G. Santarsieri](https://github.com/giuliasantarsieri)
* [P. Soriano](https://github.com/psorianom)
* [K. Montalibet](https://github.com/orgs/etalab-ia/people/KimMontalibet)

## How to contribute to this project 
We love your input! We want to make contributing to this project as easy and transparent as possible : see our [contribution rules](https://github.com/etalab-ia/ocr-xtract/blob/master/.github/contributing.md)
