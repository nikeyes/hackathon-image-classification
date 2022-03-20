import numpy as np
import requests  # to get image from the web
import streamlit as st
from PIL import Image
from scipy import misc
from skimage import io


def main():

    st.set_page_config(layout="wide")

    st.title("Clasificación de habitaciones")

    col1, col2 = st.columns(2)

    with col1:
        url = st.text_input('Pon la URL de una imagen')
        if url:
            image = io.imread(url)
            st.image(image, caption="Imagen", use_column_width=False)

    with col2:
        img_file_buffer = st.file_uploader("Carga una imagen")
        # El usuario carga una imagen
        if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))
            # st.image(image, caption="Imagen", use_column_width=False)
            st.header("Imagen Cargada")
            st.image(image, use_column_width=True, width=10)

    # El botón predicción se usa para iniciar el procesamiento
    if st.sidebar.button("Predecir..."):
        st.sidebar.success('Mejor Classe test')


if __name__ == '__main__':
    main()
