#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title
st.title("Diabetes Prediction Web App")
st.markdown("Enter patient details below to predict the risk of diabetes.")

# Input fields
preg = st.number_input('Number of Pregnancies', min_value=0)
glucose = st.number_input('Glucose Level', min_value=0)
bp = st.number_input('Blood Pressure', min_value=0)
skin = st.number_input('Skin Thickness', min_value=0)
insulin = st.number_input('Insulin Level', min_value=0)
bmi = st.number_input('BMI', min_value=0.0, format="%.1f")
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, format="%.3f")
age = st.number_input('Age', min_value=1)

# Predict button
if st.button("Predict"):
    features = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("Prediction: The person is diabetic.")
    else:
        st.success("Prediction: The person is not diabetic.")


# In[ ]:




