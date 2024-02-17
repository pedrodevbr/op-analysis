import pandas as pd
import os
import streamlit as st

# Import data 
DIR_PATH = './data/'
# op, oc, consumo, req, san, reserva
op_df = pd.read_excel(DIR_PATH+'op.XLSX')

st.button("Click me")
#st.data_editor("Edit data", op_df)
st.checkbox("I agree")
st.toggle("Enable")
st.radio("Pick one", ["cats", "dogs"])
st.selectbox("Pick one", ["cats", "dogs"])
st.multiselect("Buy", ["milk", "apples", "potatoes"])
st.slider("Pick a number", 0, 100)
st.select_slider("Pick a size", ["S", "M", "L"])
st.text_input("First name")
st.number_input("Pick a number", 0, 10)
st.text_area("Text to translate")
st.date_input("Your birthday")
st.time_input("Meeting time")
st.file_uploader("Upload a CSV")
st.camera_input("Take a picture")
st.color_picker("Pick a color")