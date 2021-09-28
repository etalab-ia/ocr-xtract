import shutil


import streamlit as st

import os
from src.image.image import RectoCNI, FeuilleDePaye
from tempfile import mkdtemp


def main():
    st.title('FDPaye-Xtractor')
    uploaded_file = st.file_uploader('Upload your file', type=['jpg', 'jpeg', 'png','pdf'])
    if uploaded_file:
        temp_folder = mkdtemp()
        with open(os.path.join(os.path.join(temp_folder, uploaded_file.name)), 'wb') as f:
            f.write(uploaded_file.getvalue())
        image = FeuilleDePaye(os.path.join(temp_folder, uploaded_file.name))
        shutil.rmtree(temp_folder) #delete temp folder
        col1, col2 = st.beta_columns(2)
        col1.header("Original")
        col1.image(image.original_image, channels='BGR')
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
            date = str(response['date']['field'])
        except:
            date = 'champ non détecté'
        try:
            entreprise = str(response['entreprise']['field'])
        except:
            entreprise = 'champ non détecté'
        try:
            somme = str(response['somme']['field'])
        except:
            somme = 'champ non détecté'

        col2.header("Extracted Information")
        col2.markdown(f'**Nom:** {nom}')
        col2.markdown(f'**Prénom:** {prenom}')
        col2.markdown(f'**Date du bulletin:** {date}')
        col2.markdown(f'**Entreprise:** {entreprise}')
        col2.markdown(f'**Sommes perçues:** {somme}')

if __name__ == '__main__':
    main()
