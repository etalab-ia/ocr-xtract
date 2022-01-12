import streamlit as st
import requests
import os

OCR_REST_API_URL = "http://192.168.210.70:5000/fdp" #os.getenv("OCR_FDP_REST_API_URL")
print(OCR_REST_API_URL)

st.title('FeuilledePaye-Xtractor')
uploaded_file = st.file_uploader('Upload your file', type=['jpg', 'jpeg', 'png', 'pdf'])
if uploaded_file:
    files = {'file': uploaded_file}
    f = requests.post(OCR_REST_API_URL, files=files)
    # TODO : write proper message when api is down
    # TODO : use the fields from scheme
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
            date = str(response['date']['field'])
        except:
            date = 'champ non détecté'
        try:
            somme = str(response['somme']['field'])
        except:
            somme = 'champ non détecté'
        try:
            entreprise = str(response['entreprise']['field'])
        except:
            entreprise = 'champ non détecté'
        st.markdown(f'**Nom:** {nom}')
        st.markdown(f'**Prénom:** {prenom}')
        st.markdown(f'**Date:** {date}')
        st.markdown(f'**Entreprise:** {entreprise}')
        st.markdown(f'**Salaire:** {somme}')