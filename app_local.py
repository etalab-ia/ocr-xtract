import shutil
from io import StringIO

import streamlit as st
import requests
import os
from src.image.image import RectoCNI
from tempfile import mkdtemp
import cv2
import numpy as np


def main():
    st.title('CNI-Xtractor')
    uploaded_file = st.file_uploader('Upload your file', type=['jpg', 'jpeg', 'png','pdf'])
    if uploaded_file:
        temp_folder = mkdtemp()
        with open(os.path.join(os.path.join(temp_folder, uploaded_file.name)), 'wb') as f:
            f.write(uploaded_file.getvalue())
        image = RectoCNI(os.path.join(temp_folder, uploaded_file.name))
        shutil.rmtree(temp_folder) #delete temp folder
        col1, col2 = st.beta_columns(2)
        col1.header("Original")
        col1.image(image.original_image, channels='BGR')
        image.align_images()
        col2.header("Aligned")
        col2.image(image.aligned_image, channels='BGR')
        response = image.extract_information()
        try:
            nom = str(response['nom']['field'])
        except:
            nom = 'champ non détecté'
        try:
            prenom = str(response['prenom']['field'])
        except:
            prenom = 'champ non détecté'
        try:
            date = str(response['date_naissance']['field'])
        except:
            date = 'champ non détecté'
        col2.header("Extracted Information")
        col2.markdown(f'**Nom:** {nom}')
        col2.markdown(f'**Prénom:** {prenom}')
        col2.markdown(f'**Date de Naissance:** {date}')

if __name__ == '__main__':
    main()
