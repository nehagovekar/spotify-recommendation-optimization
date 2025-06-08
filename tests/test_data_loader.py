import sys
import os
import pandas as pd

# Dynamically add the root directory to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from src.data_processing.data_loader import SpotifyDataLoader

def test_load_data():
    loader = SpotifyDataLoader("data/raw/Spotify_Data.csv")
    df = loader.load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    print("✅ Data loaded successfully!")

if __name__ == "__main__":
    test_load_data()
