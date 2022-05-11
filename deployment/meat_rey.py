# from os import access
from PIL import Image
from io import BytesIO
import requests
import numpy as np
import streamlit as st
import tensorflow as tf

st.set_page_config(layout="centered", page_icon="🌲🗑", page_title="Recycling Go Green ")

# Load Model
model = tf.keras.models.load_model('model_meat.h5')

# Variable for image
img = None

# Prediction
def img_predict(img):
    pred = np.asarray(img)[:, :, :3]
    pred = tf.image.resize(pred, size=(170, 170))
    pred = pred / 255.0
    prob = model.predict(x=tf.expand_dims(pred, axis=0))
    st.write(prob)
    
    labels = {
        0: 'Rotten',
        1: 'Fresh',
        }
    st.write(np.argmax(prob,axis=1)[0])        
    pred = labels[np.argmax(prob,axis=1)[0]]

    st.markdown("Setelah kami analisis, gambar yang anda unggah ")
    st.markdown(f"mempunyai kemungkinan {prob[0,np.argmax(prob)]*100:3.0f}% ")
    

    title = f"<h2 style='text-align:center'>{pred}</h2>"
    st.markdown(title, unsafe_allow_html=True)
    st.image(img, use_column_width=True)

    # st.markdown(f"and {100-prob[0,np.argmax(prob)]*100:3.0f}% is made of something else")

# Title
st.title("Halo kami akan membantu kamu dalam menentukan jenis sampah apa ini?")
st.subheader("Sini kaka kita bantu silahkan di upload atau masukan URL dari jenis sampahnya yaa ✌️")

# Image Upload Option
choose = st.selectbox("Select Input Method", ["Upload an Image", "URL from Web"])

if choose == "Upload an Image":  # If user chooses to upload image
    file = st.file_uploader("Upload an image...", type=["jpg","jpeg", "png"])
    if file is not None:
        img = Image.open(file)
else:  # If user chooses to upload image from url
    url = st.text_area("URL", placeholder="Put URL here")
    if url:
        try:  # Try to get the image from the url
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
        except:  # If the url is not valid, show error message
            st.error(
                "Coba lagi kaka, tapi kali ini menggunakan URL yang Benar ya."
            )

if img is not None:
    predict = st.button("Predict")
    if predict:
        img_predict(img)