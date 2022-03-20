# https://github.com/DavidReveloLuna/APIDeep_Streamlit
import os
import shutil  # to save it locally
import sys
import urllib.request as urllib
from io import BytesIO

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import requests  # to get image from the web
import streamlit as st
# sys.path.insert(0, '.')
# if os.environ.get('TF_KERAS'):
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
from PIL import Image
from scipy import misc
from skimage import io
from tensorflow import keras

print('tfkeras')
# else:
#    from efficientnet.keras import center_crop_and_resize, preprocess_input
#
#    print('keras')


def get_model():
    MLFLOW_TRACKING_URI = 'https://mlflow.spain-ml-dev-01.736618014913.cre.mpi-internal.com/'

    # Setup de MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    if not os.path.exists('model_downloaded'):
        print("DOWNLOAD FROM MLFLOW......")
        os.mkdir('model_downloaded')
        client.download_artifacts('c68d90e72d2f44708a51a187f7dc0d7e', 'model', 'model_downloaded')

    # logged_model = 'models:/efficientnetv2-s/24'
    # Load model as a PyFuncModel.
    model = mlflow.pyfunc.load_model('model_downloaded/model')
    return model


# Dimensiones de las imagenes de entrada
image_size = 384

# Clases
classes = ['bathroom', 'bedroom', 'dinning', 'frontal', 'kitchen', 'livingroom']


# Se recibe la imagen y el modelo, devuelve la predicción
def model_prediction(img, model):

    x = center_crop_and_resize(img, image_size=image_size)
    x = preprocess_input(x)
    x = np.expand_dims(x, 0)

    y = model.predict(x)
    return y


def main():

    st.set_page_config(layout="wide")

    if 'model' not in st.session_state:
        model = get_model()
        st.session_state['model'] = model
    else:
        model = st.session_state['model']

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
        prediction = model_prediction(image, model)
        classes_predictions = dict(zip(classes, prediction[0]))
        sorted_tuples = sorted(classes_predictions.items(), key=lambda item: item[1], reverse=True)
        st.sidebar.info(f'Predicciones: {sorted_tuples}')
        st.sidebar.success(f'Mejor Classe: {classes[np.argmax(prediction)]}')


if __name__ == '__main__':
    main()
