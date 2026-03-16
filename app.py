import streamlit as st
import pickle
import numpy as np

st.title("Smart Farm Crop Recommendation")

# Load model safely
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Example Inputs
N = st.number_input("Nitrogen")
P = st.number_input("Phosphorus")
K = st.number_input("Potassium")
temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall")

if st.button("Predict"):
    data = np.array([[N,P,K,temperature,humidity,ph,rainfall]])
    prediction = model.predict(data)
    st.success(f"Recommended Crop: {prediction[0]}")
