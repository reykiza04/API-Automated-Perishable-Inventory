import streamlit as st
import requests



def call_backend(image, endpoint):
    files = {'file': image}
    response = requests.post(endpoint, files=files).json()
    return response['class'], response['probability']


def on_show_result(_input, clazz, prob):
    st.markdown(f"<h3 style='text-align'>Setelah kami analisis, gambar yang anda unggah</h3> ", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align'>mempunyai kemungkinan {prob}% </h3>" , unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align:center'>{clazz}</h2>", unsafe_allow_html=True)
    st.image(_input, use_column_width=True)
    

def on_draw_input(endpoint):
    file = st.file_uploader("Upload an image...", type=["jpg","jpeg", "png"])
    if st.button("Predict"):
        if file:
            clazz, prob = call_backend(file, endpoint)
            on_show_result(file, clazz, prob)


def page_home():
    st.title('API: Automated Perishable Inventory')
    st.write('By: Reyki Seprianza, Ibrahim Hasan, Charissa Janto - Batch 09')
    st.write("""Objective: Tujuan dari proyek ini adalah untuk menjaga kepercayaan masyarakat 
    kepada kualitas produk di supermarket dan diharapkan akan berdampak 
    kepada kenaikan sales kedepanya.""")

def page_fruit():
    fruit_endpoint = 'https://backend-fruit-model.herokuapp.com/inference'
    st.title("🍎 Silahkan upload gambar buah yang akan diprediksi")
    on_draw_input(fruit_endpoint)


def page_meat():
    meat_endpoint = 'https://backend-meat-model.herokuapp.com/inference'
    st.title("🥩 Silahkan upload  gambar daging yang akan diprediksi")
    on_draw_input(meat_endpoint)

sidebar = st.sidebar.selectbox("Pilih Halaman", ["Home", "Model Fruit", "Model Meat"])
{
    "Home":    page_home,
    "Model Fruit":    page_fruit,
    "Model Meat":    page_meat
}[sidebar]()

