"""
Feature Engineering Pipeline for Spotify Hit Prediction
Create smart features that help predict song success
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

class SpotifyFeatureEngineer:
    """Create features that predict song success"""
    
    def __init__(self):
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
        
        created_features = []
        for feat1, feat2, name in interactions:
            if feat1 in df.columns and feat2 in df.columns:
                df_features[name] = df_features[feat1] * df_features[feat2]
                created_features.append(name)
                print(f"   ✅ Created {name}")
        
        self.feature_names.extend(created_features)
        return df_features
    
    def create_composite_scores(self, df):
        """Create composite scores that summarize multiple features"""
        print("🔄 Creating composite scores...")
        
        df_features = df.copy()
        created_features = []
        
        # Happiness Score (valence + danceability - acousticness)
        if all(f in df.columns for f in ['valence', 'danceability', 'acousticness']):
            df_features['happiness_score'] = (
                df_features['valence'] + 
                df_features['danceability'] - 
                df_features['acousticness']
            ) / 2
            created_features.append('happiness_score')
            print("   ✅ Created happiness_score")
        
        # Dancefloor Potential (danceability + energy + tempo_normalized)
        if all(f in df.columns for f in ['danceability', 'energy', 'tempo']):
            # Normalize tempo to 0-1 range
            tempo_norm = (df_features['tempo'] - df_features['tempo'].min()) / (
                df_features['tempo'].max() - df_features['tempo'].min()
            )
            df_features['dancefloor_potential'] = (
                df_features['danceability'] + 
                df_features['energy'] + 
                tempo_norm
            ) / 3
            created_features.append('dancefloor_potential')
            print("   ✅ Created dancefloor_potential")
        
        # Chill Factor (acousticness + (1-energy) + (1-loudness_normalized))
        if all(f in df.columns for f in ['acousticness', 'energy', 'loudness']):
            # Normalize loudness (usually negative values)
            loudness_norm = (df_features['loudness'] - df_features['loudness'].min()) / (
                df_features['loudness'].max() - df_features['loudness'].min()
            )
            df_features['chill_factor'] = (
                df_features['acousticness'] + 
                (1 - df_features['energy']) + 
                (1 - loudness_norm)
            ) / 3
            created_features.append('chill_factor')
            print("   ✅ Created chill_factor")
        
        self.feature_names.extend(created_features)
        return df_features
    
    def create_categorical_features(self, df):
        """Create categorical features from continuous ones"""
        print("🔄 Creating categorical features...")
        
        df_features = df.copy()
        created_features = []
        
        # Energy levels
        if 'energy' in df.columns:
            df_features['energy_level'] = pd.cut(
                df_features['energy'], 
                bins=[0, 0.3, 0.7, 1.0], 
                labels=['Low', 'Medium', 'High']
            )
            created_features.append('energy_level')
            print("   ✅ Created energy_level (Low/Medium/High)")
        
        # Danceability categories
        if 'danceability' in df.columns:
            df_features['dance_category'] = pd.cut(
                df_features['danceability'], 
                bins=[0, 0.4, 0.7, 1.0], 
                labels=['Not_Danceable', 'Moderate', 'Very_Danceable']
            )
            created_features.append('dance_category')
            print("   ✅ Created dance_category")
        
        # Tempo categories
        if 'tempo' in df.columns:
            df_features['tempo_category'] = pd.cut(
                df_features['tempo'], 
                bins=[0, 90, 120, 140, 200], 
                labels=['Slow', 'Medium', 'Fast', 'Very_Fast']
            )
            created_features.append('tempo_category')
            print("   ✅ Created tempo_category")
        
        # Valence mood
        if 'valence' in df.columns:
            df_features['mood'] = pd.cut(
                df_features['valence'], 
                bins=[0, 0.33, 0.67, 1.0], 
                labels=['Sad', 'Neutral', 'Happy']
            )
            created_features.append('mood')
            print("   ✅ Created mood (Sad/Neutral/Happy)")
        
        self.feature_names.extend(created_features)
        return df_features
    
    def create_ratio_features(self, df):
        """Create ratio features"""
        print("🔄 Creating ratio features...")
        
        df_features = df.copy()
        created_features = []
        
        # Speech to music ratio
        if all(f in df.columns for f in ['speechiness', 'instrumentalness']):
            df_features['speech_to_music_ratio'] = (
                df_features['speechiness'] / 
                (df_features['instrumentalness'] + 0.001)  # Avoid division by zero
            )
            created_features.append('speech_to_music_ratio')
            print("   ✅ Created speech_to_music_ratio")
        
        # Energy to acousticness ratio
        if all(f in df.columns for f in ['energy', 'acousticness']):
            df_features['energy_acoustic_ratio'] = (
                df_features['energy'] / 
                (df_features['acousticness'] + 0.001)
            )
            created_features.append('energy_acoustic_ratio')
            print("   ✅ Created energy_acoustic_ratio")
        
        self.feature_names.extend(created_features)
        return df_features
    
    def create_all_features(self, df):
        """Create all engineered features"""
        print("\n🚀 STARTING FEATURE ENGINEERING")
        print("="*50)
        
        original_features = len(df.columns)
        
        # Apply all feature engineering steps
        df_engineered = df.copy()
        df_engineered = self.create_interaction_features(df_engineered)
        df_engineered = self.create_composite_scores(df_engineered)
        df_engineered = self.create_categorical_features(df_engineered)
        df_engineered = self.create_ratio_features(df_engineered)
        
        new_features = len(df_engineered.columns) - original_features
        
        print(f"\n✅ FEATURE ENGINEERING COMPLETE!")
        print(f"   Original features: {original_features}")
        print(f"   New features created: {new_features}")
        print(f"   Total features: {len(df_engineered.columns)}")
        
        return df_engineered
    
    def get_feature_importance_preview(self, df):
        """Quick preview of feature relationships with target"""
        if 'target' not in df.columns:
            print("⚠️ No target column found for feature importance preview")
            return
        
        print("\n🔍 FEATURE IMPORTANCE PREVIEW:")
        print("="*40)
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'target']
        
        # Calculate correlations with target
        correlations = df[numeric_cols + ['target']].corr()['target'].abs().sort_values(ascending=False)
        
        print("Top features correlated with success:")
        for feature, corr in correlations.head(10).items():
            if feature != 'target':
                print(f"   {feature:25s}: {corr:.3f}")

def main():
    """Run feature engineering pipeline"""
    
    print("🔧 SPOTIFY FEATURE ENGINEERING PIPELINE")
    print("="*50)
    
    # Load data
    data_files = [
        "data/processed/spotify_clean.csv",
        "data/raw/Spotify_Data.csv",
        "data/raw/spotify_songs.csv"
    ]
    
    df = None
    for file_path in data_files:
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            print(f"✅ Loaded data from: {file_path}")
            break
    
    if df is None:
        print("❌ No data file found!")
        print("Available files:")
        for path in Path("data").rglob("*.csv"):
            print(f"   {path}")
        return
    
    # Create feature engineer
    engineer = SpotifyFeatureEngineer()
    
    # Engineer features
    df_engineered = engineer.create_all_features(df)
    
    # Preview feature importance
    engineer.get_feature_importance_preview(df_engineered)
    
    # Save engineered features
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "spotify_features_engineered.csv"
    df_engineered.to_csv(output_path, index=False)
    
    print(f"\n💾 FEATURES SAVED:")
    print(f"   Location: {output_path}")
    print(f"   Shape: {df_engineered.shape}")
    
    # Summary of created features
    print(f"\n📋 NEW FEATURES CREATED:")
    for feature in engineer.feature_names:
        print(f"   • {feature}")
    
    print(f"\n🚀 READY FOR MACHINE LEARNING!")
    print("Next steps:")
    print("1. Train ML models with engineered features")
    print("2. Compare model performance")
    print("3. Set up A/B testing framework")
    
    return df_engineered

if __name__ == "__main__":
    df_engineered = main()