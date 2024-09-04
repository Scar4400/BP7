import pandas as pd
import numpy as np

def preprocess_pinnacle_data(data):
    df = pd.DataFrame(data["special_markets"])
    df["timestamp"] = pd.to_datetime(df["updated"])
    return df

def preprocess_livescore_data(data):
    df = pd.DataFrame(data["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df

def preprocess_api_football_data(data):
    df = pd.DataFrame(data["api"]["odds"])
    df["timestamp"] = pd.to_datetime(df["update"], unit="s")
    return df

def merge_data(pinnacle_df, livescore_df, api_football_df):
    # Merge dataframes based on common fields (e.g., team names, match dates)
    # This is a simplified example and may need to be adjusted based on the actual data structure
    merged_df = pd.merge(pinnacle_df, livescore_df, on=["team_name", "timestamp"], how="outer")
    merged_df = pd.merge(merged_df, api_football_df, on=["team_name", "timestamp"], how="outer")

    # Handle missing values
    merged_df = merged_df.fillna(method="ffill").fillna(method="bfill")

    return merged_df

def preprocess_data(data):
    pinnacle_df = preprocess_pinnacle_data(data["pinnacle"])
    livescore_df = preprocess_livescore_data(data["livescore"])
    api_football_df = preprocess_api_football_data(data["api_football"])

    merged_df = merge_data(pinnacle_df, livescore_df, api_football_df)

    # Remove duplicates
    merged_df = merged_df.drop_duplicates()

    # Sort by timestamp
    merged_df = merged_df.sort_values("timestamp")

    return merged_df

if __name__ == "__main__":
    # For testing purposes
    import json

    with open("sample_data.json", "r") as f:
        sample_data = json.load(f)

    processed_data = preprocess_data(sample_data)
    print(processed_data.head())
    print(processed_data.info())

