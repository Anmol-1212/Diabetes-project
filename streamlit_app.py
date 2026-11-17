
import streamlit as st
import numpy as np
import pickle

# Load trained pipeline
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

from pathlib import Path

st.write("Model location:", Path("diabetes_model.pkl").resolve())
st.write("Model type:", type(model).__name__)

st.title("Diabetes Prediction Web App")

st.write("Enter patient details below.")

preg = st.number_input("Number of Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin Level", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=1)

if st.button("Predict"):

    st.write("Input sent to model:", features)
    st.write("Raw prediction:", prediction)

    features = np.array([[
        preg, glucose, bp, skin,
        insulin, bmi, dpf, age
    ]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("Prediction: Higher likelihood of diabetes.")
    else:
        st.success("Prediction: Lower likelihood of diabetes.")

    st.caption(
        "This is an ML model prediction, not a medical diagnosis."
    )
