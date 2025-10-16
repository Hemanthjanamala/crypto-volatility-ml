# File: src/data/split.py

import pandas as pd
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

def time_series_split(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Split dataset into train/test by time order (not random shuffle).
    """
    df = df.sort_values(["Name","Date"])
    
    split_index = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    # Save split indices for reproducibility
    split_meta = {
        "train_size": len(train_df),
        "test_size": len(test_df),
        "split_index": split_index
    }
    
    Path("data/processed/").mkdir(parents=True, exist_ok=True)
    with open("data/processed/split_indices.json", "w") as f:
        json.dump(split_meta, f, indent=4)
    
    return train_df, test_df
