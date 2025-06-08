"""
Feature Engineering for Spotify Hit Prediction
Create smart features that help ML models predict song success
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class SpotifyFeatureEngineer:
    """Create features that predict song success"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Define audio features
        self.audio_features = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
        ]
    
    def create_interaction_features(self, df):
        """Create interaction features that combine audio characteristics"""
        print("🔄 Creating interaction features...")
        
        df_features = df.copy()
        
        # Key interactions that often predict hit songs
        interactions = [
            ('energy', 'danceability', 'energy_dance'),
            ('valence', 'danceability', 'happy_dance'),
            ('energy', 'loudness', 'energy_loudness'),
            ('acousticness', 'energy', 'acoustic_energy'),
            ('danceability', 'tempo', 'dance_tempo')
        ]
        
        for feat1, feat2, name in interactions:
            if feat1 in df.columns and feat2 in df.columns:
                df_features[name] = df_features[feat1] * df_features[feat2]
                print(f"   ✅ Created {name}")
        
        return df_features
    
    def create_composite_scores(self, df):
        """Create composite scores that summarize multiple features"""