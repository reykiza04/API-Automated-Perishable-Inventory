from turtle import home
import fruit
import MeatOLD
import Home
import streamlit as st


st.set_page_config(
    page_title="API: Automated Perishable Inventory",
    page_icon="🍔")
    

PAGES = {
    "Home": Home,
    "Meat Detection": MeatOLD,
    "Fruit Detection": fruit
}

###### Multi Pages
st.sidebar.title('Navigation')
selection = st.sidebar.selectbox("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()