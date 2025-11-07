
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model


# Load model and preprocessors
model = load_model("ipl_score_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

st.set_page_config(page_title="🏏 IPL Score Predictor", layout="centered")

st.title("🏏 IPL Score Prediction App")
st.markdown("### Predict the total score based on the current match situation")

# Dropdowns
venue = st.selectbox("Venue", label_encoders['venue'].classes_)
bat_team = st.selectbox("Batting Team", label_encoders['bat_team'].classes_)
bowl_team = st.selectbox("Bowling Team", label_encoders['bowl_team'].classes_)
batsman = st.selectbox("Striker", label_encoders['batsman'].classes_)
bowler = st.selectbox("Bowler", label_encoders['bowler'].classes_)

# Numerical inputs
runs = st.number_input("Current Runs", 0, 300, 50)
wickets = st.number_input("Wickets", 0, 10, 2)
overs = st.number_input("Overs Completed", 0.0, 20.0, 10.0, step=0.1)
runs_last_5 = st.number_input("Runs in Last 5 Overs", 0, 100, 30)
wickets_last_5 = st.number_input("Wickets in Last 5 Overs", 0, 5, 1)

# Derived features
current_run_rate = 0 if overs == 0 else runs / overs
remaining_overs = 20 - overs
is_powerplay = 1 if overs <= 6 else 0
run_rate_last_5 = runs_last_5 / 5

# Convert all categorical data using saved encoders
encoded_features = [
    label_encoders['venue'].transform([venue])[0],
    label_encoders['bat_team'].transform([bat_team])[0],
    label_encoders['bowl_team'].transform([bowl_team])[0],
    label_encoders['batsman'].transform([batsman])[0],
    label_encoders['bowler'].transform([bowler])[0],
    runs, wickets, overs, runs_last_5, wickets_last_5,
    current_run_rate, remaining_overs, is_powerplay, run_rate_last_5,
    0, 0  # Placeholder for striker, non-striker (not used)
]

# Scale the input
input_scaled = scaler.transform(np.array(encoded_features).reshape(1, -1))

if st.button("Predict Score"):
    predicted = model.predict(input_scaled)
    st.success(f"🎯 Predicted Total Score: {int(predicted[0][0])} runs")
