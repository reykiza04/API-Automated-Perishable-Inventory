from os import access
import re
from PIL import Image
import requests
import numpy as np
import streamlit as st
import tensorflow as tf

def app():
    # st.set_page_config(layout="centered", page_title="(re)tire ?")
    labels = { 0:'Fresh', 1:'Tidak Fresh' }

    # Load Model
    model = tf.keras.models.load_model('model_meat.h5')

    # Variable for image
    img = None

    # Prediction
    def img_predict(img):
        pred = np.array(img)[:,:,:3]
        pred = tf.image.resize(pred, size=(170, 170))
        pred = pred / 255.0

        
        prob = model.predict(x=tf.expand_dims(pred, axis=0))
        res=labels[int(tf.round(prob))]

        
        # res = labels[int(tf.round(model.predict(x=tf.expand_dims(pred, axis=0))))]

        if res == 'Fresh':
            st.subheader(f"Produk ini")
            title = f"<h2 style='text-align:center'>{res}</h2>"
            st.markdown(title, unsafe_allow_html=True)
        else:
            st.subheader("Tidak boleh di jual, produk ini")
            title = f"<h2 style='text-align:center'>{res}</h2>"
            st.markdown(title, unsafe_allow_html=True)

        # st.markdown(f"<h2 style='text-align:center'>mempunyai kemungkinan {prob[0,np.argmax(prob)]*100:3.0f}% </h2>",unsafe_allow_html=True)
        st.image(img, use_column_width=True)

    # Title
    st.title("🥩 Silahkan upload  gambar daging yang akan diprediksi")

    # Image Upload Option

    file = st.file_uploader("Upload an image", type=["jpg","jpeg", "png"])

    if file is not None:
        img = Image.open(file)

    if img is not None:
        predict = st.button("Check")
        if predict:
            img_predict(img)