import streamlit as st
import requests
import os

OCR_REST_API_URL = os.getenv("OCR_CNI_REST_API_URL")

st.title('CNI-Xtractor')
uploaded_file = st.file_uploader('Upload your file', type=['jpg', 'jpeg', 'png', 'pdf'])
if uploaded_file:
    files = {'file': uploaded_file}
    print(OCR_REST_API_URL)
    f = requests.post(OCR_REST_API_URL, files=files)
    # TODO : write proper message when api is down
    if f:
        response = f.json()
        response = response['result']
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
        st.markdown(f'**Nom:** {nom}')
        st.markdown(f'**Prénom:** {prenom}')
        st.markdown(f'**Date de Naissance:** {date}')