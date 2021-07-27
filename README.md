# OCR_XTRACT

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

- OCR
- Image Processing

### Technologies

- Python

## Project Description

For now, only a POC is avaible for extracting information for French DNI

## Getting Started for development

- Fork this repo
- Install requirements : `pip install -r requirements.txt`

Since we use [doctr](https://mindee.github.io/doctr/), you will need extra dependencies if you are not running Linux.
For MacOS users, you can install them as follows:

```shell
brew install cairo pango gdk-pixbuf libffi
```

For Windows users, those dependencies are included in GTK. You can find the latest installer over [here](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases).

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
python -m streamlit.cli run front/app_local.py
```

## Contributing Lab IA Members

- [R. Reynaud](https://github.com/rob192)
- [G. Santarsieri](https://github.com/giuliasantarsieri)
- [P. Soriano](https://github.com/psorianom)
- [K. Montalibet](https://github.com/KimMontalibet)

## How to contribute to this project

We love your input! We want to make contributing to this project as easy and transparent as possible : see our [contribution rules](https://github.com/etalab-ia/ocr-xtract/blob/master/.github/contributing.md)
