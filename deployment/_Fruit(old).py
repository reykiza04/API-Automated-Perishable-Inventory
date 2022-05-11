from os import access
from PIL import Image
from io import BytesIO
import requests
import numpy as np
import streamlit as st
import tensorflow as tf

def app():
    # st.set_page_config(layout="centered", page_title="(re)tire ?")
    labels = { 0:'fresh', 1:'rotten'}

    # Load Model
    model = tf.keras.models.load_model('model_fruit.h5')

    # Variable for image
    img = None

    # Prediction
    def img_predict(img):
        pred = np.array(img)[:,:,:3]
        pred = tf.image.resize(pred, size=(210, 210))
        pred = pred / 255.0

        res = labels[int(tf.round(model.predict(x=tf.expand_dims(pred, axis=0)[0])))]

        if res == 'fresh':
            st.subheader(f"This item is fresh")
        else:
            st.subheader("dont sell it")

        title = f"<h2 style='text-align:center'>{res}</h2>"
        st.markdown(title, unsafe_allow_html=True)
        st.image(img, use_column_width=True)

    # Title
    st.title("Do not let your tire in tear")
    st.subheader("Upload your tire image for checking here")

    # Image Upload Option

    file = st.file_uploader("Upload an image", type=["jpg","jpeg", "png"])

    if file is not None:
        img = Image.open(file)

    if img is not None:
        predict = st.button("Check")
        if predict:
            img_predict(img)
            