from io import StringIO

import streamlit as st
import requests
import os
from src.image.image import RectoCNI
from tempfile import mkdtemp
import cv2
import numpy as np
url = "http://127.0.0.1:8000/"

st.title('CNI-Xtractor')
uploaded_file = st.file_uploader('Upload your file', type=['jpg', 'jpeg', 'png'])
if uploaded_file:
    temp_folder = mkdtemp()
    with open(os.path.join(os.path.join(temp_folder, 'temp.jpg')), 'wb') as f:
        f.write(uploaded_file.getvalue())
    image = RectoCNI(os.path.join(temp_folder, 'temp.jpg'))
    col1, col2 = st.beta_columns(2)
    col1.header("Original")
    col1.image(image.original_image)
    image.align_images()
    col2.header("Aligned")
    col2.image(image.aligned_image)
    response = image.extract_information()
    nom = str(response['nom']['field'])
    prenom = str(response['prenom']['field'])
    date = str(response['date_naissance']['field'])
    col2.header("Extracted Information")
    col2.markdown(f'**Nom:** {nom}')
    col2.markdown(f'**Prénom:** {prenom}')
    col2.markdown(f'**Date de Naissance:** {date}')