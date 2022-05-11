from os import access
from PIL import Image
from io import BytesIO
import requests
import numpy as np
import streamlit as st
import tensorflow as tf

st.set_page_config(layout="centered", page_icon="🗑", page_title="Recycling Facility Sorting")

# Load Model
model = tf.keras.models.load_model('best_model.h5')

# Variable for image
img = None

# Prediction
def img_predict(img):
    pred = np.array(img)
    pred = tf.image.resize(pred, size=(128, 128))
    pred = pred / 255.0

    prob = model.predict(x=tf.expand_dims(pred, axis=0))

    
    labels = { 0:'cardboard', 1:'glass', 2:'metal', 3:'paper' , 4:'plastic', 5:'trash'}
    pred = labels[np.argmax(prob)]

    st.markdown("Yo! im pretty sureee...")
    st.markdown(f"Your garbage is{prob[0,np.argmax(prob)]*100:3.0f}% made of")
    

    title = f"<h2 style='text-align:center'>{pred}</h2>"
    st.markdown(title, unsafe_allow_html=True)
    st.image(img, use_column_width=True)

    st.markdown(f"and {100-prob[0,np.argmax(prob)]*100:3.0f}% is made of something else")

# Title
st.title("Sup?! Got a garbage problem?")
st.subheader("Yo! let me sort out your garbage ✌️")

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
                "Failed to load the image. Please use a different URL or upload an image."
            )

if img is not None:
    predict = st.button("Predict")
    if predict:
        img_predict(img)