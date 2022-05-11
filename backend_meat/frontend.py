from asyncore import write
from urllib import response
from PIL import Image
import numpy as np
import streamlit as st
import requests
from io import BytesIO

backend_url = 'http://localhost:5000/inference'

def call_backend(image):
    files = {'file': image}
    response = requests.post(backend_url, files=files).json()
    return response


def inference(img):
    response = call_backend(img)
    clazz = response['class']
    prob = response['probability']

    st.markdown(f"<h3 style='text-align'>Setelah kami analisis, gambar yang anda unggah</h3> ", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align'>mempunyai kemungkinan {prob}% </h3>" , unsafe_allow_html=True)


    title = f"<h2 style='text-align:center'>{clazz}</h2>"
    st.markdown(title, unsafe_allow_html=True)
    st.image(img, use_column_width=True)

# Title
st.title("🍎 Silahkan upload gambar buah yang akan diprediksi")

# Image Upload Option
img = None
choose = st.selectbox("Select Input Method", ["Upload an Image", "URL from Web"])

file = st.file_uploader("Upload an image...", type=["jpg","jpeg", "png"])
if st.button("Predict"):
    if file:
        inference(file)