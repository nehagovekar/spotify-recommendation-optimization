import os
import sys
import pandas as pd

# Allow access to src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data_processing.data_loader import SpotifyDataLoader

def test_load_data():
    loader = SpotifyDataLoader()
    df = loader.load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    print("[SUCCESS] Test passed: DataFrame loaded.")

if __name__ == "__main__":
    test_load_data()
