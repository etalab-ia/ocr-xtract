import shutil
import os
from io import StringIO

import streamlit as st

from src.image.image import RectoCNI
from tempfile import mkdtemp


def process_image(file):
    temp_folder = mkdtemp()

    with open(os.path.join(os.path.join(temp_folder, "temp.jpg")), "wb") as f:
        f.write(file.getvalue())

    image = RectoCNI(os.path.join(temp_folder, "temp.jpg"))
    image.align_images()

    shutil.rmtree(temp_folder)

    return image.extract_information(), image


def display(response, processed_image):
    default_label = "champ non détecté"

    nom = str(response["nom"]["field"]) or default_label
    prenom = str(response["prenom"]["field"]) or default_label
    date = str(response["date_naissance"]["field"]) or default_label

    col1, col2 = st.beta_columns(2)
    col1.header("Original")
    col1.image(processed_image.original_image)
    col2.header("Aligned")
    col2.image(processed_image.aligned_image)

    col2.header("Extracted Information")
    col2.markdown(f"**Nom:** {nom}")
    col2.markdown(f"**Prénom:** {prenom}")
    col2.markdown(f"**Date de Naissance:** {date}")


def main():
    st.title("CNI-Xtractor")

    types = ["jpg", "jpeg", "png"]

    uploaded_file = st.file_uploader("Upload your file", type=types)

    if uploaded_file:
        response, processed_image = process_image(uploaded_file)
        display(response, processed_image)


if __name__ == "__main__":
    main()
