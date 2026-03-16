import streamlit as st
import numpy as np
import joblib

st.title("Smart Farm Crop Recommendation System")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

st.header("Enter Soil and Weather Details")

N = st.number_input("Nitrogen", min_value=0.0)
P = st.number_input("Phosphorus", min_value=0.0)
K = st.number_input("Potassium", min_value=0.0)
temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall")

if st.button("Predict Crop"):
    
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    data = scaler.transform(data)
    
    prediction = model.predict(data)

    st.success(f"Recommended Crop: {prediction[0]}")
