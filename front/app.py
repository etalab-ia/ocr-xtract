import streamlit as st
import requests

url = "http://127.0.0.1:8000/"

st.title('CNI-Xtractor')
uploaded_file = st.file_uploader('Upload your file', type=['jpg', 'jpeg', 'png'])
if uploaded_file:
    files = {'file': uploaded_file}
    f = requests.post(url, files=files)
    if f:
        response = f.json()
        response = response['result']['RectoCNI'][0]
        nom = response['nom']
        prenom = response['prenom']
        date = response['date_de_naissance']
        st.markdown(f'**Nom:** {nom}')
        st.markdown(f'**Prénom:** {prenom}')
        st.markdown(f'**Date de Naissance:** {date}')