import mlflow
import numpy as np
import streamlit as st
from PIL import Image
from skimage import io


def get_model():
    model = mlflow.pyfunc.load_model('model_downloaded')
    return model


# Clases
classes = ['bathroom', 'bedroom', 'dinning', 'frontal', 'kitchen', 'livingroom']


# Se recibe la imagen y el modelo, devuelve la predicción
def model_prediction(image, model):
    x = np.expand_dims(image, axis=0)

    y = model.predict(x)
    return y


def main():

    st.set_page_config(layout="wide")

    if 'model' not in st.session_state:
        model = get_model()
        st.session_state['model'] = model
    else:
        model = st.session_state['model']

    st.title("Clasificador de habitaciones")

    col1, col2, col3 = st.columns(3)

    with col2:
        url = st.text_input('Pon la URL de una imagen')
        if url:
            image = io.imread(url)
            st.image(image, caption="Imagen", use_column_width=False)

    with col3:
        img_file_buffer = st.file_uploader("Carga una imagen")
        # El usuario carga una imagen
        if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))
            # st.image(image, caption="Imagen", use_column_width=False)
            st.header("Imagen Cargada")
            st.image(image, use_column_width=True, width=10)

    with col1:
        # El botón predicción se usa para iniciar el procesamiento
        if st.button("Predecir..."):
            prediction = model_prediction(image, model)
            classes_predictions = dict(zip(classes, prediction[0]))
            sorted_tuples = sorted(classes_predictions.items(), key=lambda item: item[1], reverse=True)
            st.info(f'Predicciones: {sorted_tuples}')
            st.success(f'Mejor Classe: {classes[np.argmax(prediction)]}')


if __name__ == '__main__':
    main()
