"""
Test Feature Engineering Pipeline
Quick test to make sure everything works
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

def create_sample_data_with_target():
    """Create sample data for testing"""
    print("🎵 Creating sample data for testing...")
    
    np.random.seed(42)
    n_songs = 500
    
    # Create realistic audio features
    data = {
        'acousticness': np.random.beta(2, 5, n_songs),
        'danceability': np.random.beta(2, 2, n_songs),
        'energy': np.random.beta(2, 2, n_songs),
        'instrumentalness': np.random.beta(1, 10, n_songs),
        'liveness': np.random.beta(1, 9, n_songs),
        'loudness': np.random.normal(-8, 4, n_songs),
        'speechiness': np.random.beta(1, 10, n_songs),
        'tempo': np.random.normal(120, 30, n_songs),
        'valence': np.random.beta(2, 2, n_songs),
    }
    
    df = pd.DataFrame(data)
    
    # Create target based on features (songs with high energy + danceability more likely to be hits)
    hit_probability = (
        df['energy'] * 0.3 +
        df['danceability'] * 0.3 +
        df['valence'] * 0.2 +
        (1 - df['acousticness']) * 0.2
    )
    
    # Add some randomness
    hit_probability += np.random.normal(0, 0.1, n_songs)
    hit_probability = np.clip(hit_probability, 0, 1)
    
    # Convert to binary target
    df['target'] = (hit_probability > hit_probability.median()).astype(int)
    
    print(f"✅ Created {len(df)} songs with {df['target'].mean():.1%} hit rate")
    
    # Save for testing
    test_dir = Path("data/test")
    test_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(test_dir / "test_data.csv", index=False)
    
    return df

def test_feature_engineering():
    """Test the feature engineering pipeline"""
    
    print("🧪 TESTING FEATURE ENGINEERING")
    print("="*40)
    
    # Check if we have real data, otherwise create test data
    data_files = [
        "data/processed/spotify_clean.csv",
        "data/raw/Spotify_Data.csv"
    ]
    
    df = None
    for file_path in data_files:
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            print(f"✅ Using real data from: {file_path}")
            break
    
    if df is None:
        print("📁 No real data found, creating test data...")
        df = create_sample_data_with_target()
    
    # Import our feature engineering
    try:
        sys.path.append('scripts')
        from create_features import SpotifyFeatureEngineer
        #from scripts.create_features import SpotifyFeatureEngineer
        print("✅ Successfully imported SpotifyFeatureEngineer")
    except ImportError:
        print("❌ Could not import feature engineer")
        return
    
    # Test feature engineering
    engineer = SpotifyFeatureEngineer()
    
    print(f"\n📊 Original data shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Test each step
    print(f"\n1️⃣ Testing interaction features...")
    df_step1 = engineer.create_interaction_features(df)
    
    print(f"\n2️⃣ Testing composite scores...")
    df_step2 = engineer.create_composite_scores(df_step1)
    
    print(f"\n3️⃣ Testing categorical features...")
    df_step3 = engineer.create_categorical_features(df_step2)
    
    print(f"\n4️⃣ Testing ratio features...")
    df_final = engineer.create_ratio_features(df_step3)
    
    print(f"\n📈 Final data shape: {df_final.shape}")
    print(f"Features added: {df_final.shape[1] - df.shape[1]}")
    
    # Show new features
    new_features = [col for col in df_final.columns if col not in df.columns]
    print(f"\n🆕 NEW FEATURES CREATED:")
    for feature in new_features:
        print(f"   • {feature}")
    
    # Test feature importance preview
    if 'target' in df_final.columns:
        engineer.get_feature_importance_preview(df_final)
    
    print(f"\n✅ FEATURE ENGINEERING TEST PASSED!")
    return df_final

if __name__ == "__main__":
    df_engineered = test_feature_engineering()
    
    if df_engineered is not None:
        print(f"\n🚀 SUCCESS! Ready for machine learning with {df_engineered.shape[1]} features!")