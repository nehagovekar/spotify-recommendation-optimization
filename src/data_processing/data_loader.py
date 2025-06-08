import pandas as pd
import yaml
import os

class SpotifyDataLoader:
    def __init__(self, config_path: str = "config/data_config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.data_path = config["data_path"]

    def load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.data_path)
            print(f"[INFO] Data loaded successfully from {self.data_path}")
            return df
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            raise
