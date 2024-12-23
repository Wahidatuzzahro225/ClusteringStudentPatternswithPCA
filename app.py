import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model K-Means
model = joblib.load('kmeans_modelpca_studentfix3.pkl')
pca = joblib.load('pca_model.pkl')

# Judul aplikasi
st.title("Prediksi Cluster Student Sleep dengan K-Means")
st.write("Gunakan slider di bawah untuk memasukkan data baru dan prediksi cluster.")

# Form input
with st.form('prediction'):
    Age = st.slider("Age (years)", min_value=18.0, max_value=25.0, step=0.1, value=20.0)
    Gender = st.select_slider("Gender", options=["Male", "Female", "Other"])
    Gender_encoded = 0 if Gender == "Male" else 1 if Gender == "Female" else 2
    University_Year = st.slider("University Year (1-4)", min_value=1, max_value=4, step=1, value=3)
    Sleep_Duration = st.slider("Sleep Duration (hours)", min_value=4.0, max_value=9.0, step=0.1, value=7.0)
    Study_Hours = st.slider("Study Hours (hours/day)", min_value=0.1, max_value=12.0, step=0.1, value=5.0)
    Screen_Time = st.slider("Screen Time (hours/day)", min_value=1.0, max_value=12.0, step=0.1, value=3.0)
    Caffeine_Intake = st.slider("Caffeine Intake (cups/day)", min_value=0, max_value=4, step=1, value=2)
    Physical_Activity = st.slider("Physical Activity (hours/week)", min_value=0.0, max_value=120.0, step=0.1, value=3.5)
    Sleep_Quality = st.slider("Sleep Quality (1-10)", min_value=1, max_value=10, step=1, value=3)
    Weekday_Sleep_Start = st.slider("Weekday Sleep Start (hour)", min_value=1, max_value=24, step=1, value=23)
    Weekend_Sleep_Start = st.slider("Weekend Sleep Start (hour)", min_value=1, max_value=24, step=1, value=23)
    Weekday_Sleep_End = st.slider("Weekday Sleep End (hour)", min_value=1, max_value=24, step=1, value=8)
    Weekend_Sleep_End = st.slider("Weekend Sleep End (hour)", min_value=1, max_value=24, step=1, value=8)
    
    # Submit button
    submit_button = st.form_submit_button("Submit")

# Process input when form is submitted
if submit_button:
    new_data = np.array([[
        Age,
        Sleep_Duration, 
        University_Year, 
        Study_Hours, 
        Screen_Time, 
        Caffeine_Intake, 
        Physical_Activity, 
        Sleep_Quality, 
    ]])

    X_pca = pca.transform(new_data)

    predicted_cluster = model.predict(X_pca)
    st.subheader("Hasil Prediksi:")
    st.write(f"Data baru masuk ke cluster: *{predicted_cluster[0]}*")