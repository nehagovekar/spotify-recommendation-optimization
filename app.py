import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.title("🎵 Spotify Hit Predictor")

# Try different path combinations
possible_paths = [
    'spotify-recommendation-optimization/data/processed/spotify_features_engineered.csv',
    'data/processed/spotify_features_engineered.csv',
    '../data/processed/spotify_features_engineered.csv'
]

df = None
for path in possible_paths:
    if Path(path).exists():
        df = pd.read_csv(path)
        st.success(f"✅ Data loaded from: {path}")
        break

if df is None:
    st.error("❌ Could not find data file")
    st.stop()

# Simple prediction interface
st.subheader("🎯 Predict Song Success")

danceability = st.slider("🕺 Danceability", 0.0, 1.0, 0.5)
energy = st.slider("⚡ Energy", 0.0, 1.0, 0.5)
valence = st.slider("😊 Happiness", 0.0, 1.0, 0.5)

# Simple rule-based prediction for now
hit_score = (danceability + energy + valence) / 3

if st.button("🎯 Predict Hit"):
    if hit_score > 0.6:
        st.success(f"🎉 POTENTIAL HIT! Score: {hit_score:.1%}")
        st.balloons()
    else:
        st.info(f"📊 Needs Work. Score: {hit_score:.1%}")

st.write(f"Dataset: {len(df)} songs loaded")