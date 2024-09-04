import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def calculate_rolling_averages(df, columns, window=5):
    for col in columns:
        df[f"{col}_rolling_avg"] = df.groupby("team_name")[col].rolling(window=window).mean().reset_index(0, drop=True)
    return df

def calculate_odds_ratios(df):
    df["home_odds_ratio"] = df["home_odds"] / (df["home_odds"] + df["away_odds"])
    df["away_odds_ratio"] = df["away_odds"] / (df["home_odds"] + df["away_odds"])
    return df

def encode_categorical_features(df):
    categorical_columns = ["league", "team_name", "opponent_name"]
    return pd.get_dummies(df, columns=categorical_columns, drop_first=True)

def create_time_features(df):
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df

def calculate_team_form(df, window=5):
    df["points"] = np.where(df["result"] == "W", 3, np.where(df["result"] == "D", 1, 0))
    df["form"] = df.groupby("team_name")["points"].rolling(window=window).sum().reset_index(0, drop=True)
    return df

def create_features(df):
    # Calculate rolling averages for relevant columns
    rolling_columns = ["goals_scored", "goals_conceded", "shots_on_target", "possession"]
    df = calculate_rolling_averages(df, rolling_columns)

    # Calculate odds ratios
    df = calculate_odds_ratios(df)

    # Encode categorical features
    df = encode_categorical_features(df)

    # Create time-based features
    df = create_time_features(df)

    # Calculate team form
    df = calculate_team_form(df)

    # Create interaction features
    df["form_odds_interaction"] = df["form"] * df["home_odds_ratio"]

    # Scale numerical features
    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df

if __name__ == "__main__":
    # For testing purposes
    import json
    from data_preprocessing import preprocess_data

    with open("sample_data.json", "r") as f:
        sample_data = json.load(f)

    processed_data = preprocess_data(sample_data)
    featured_data = create_features(processed_data)
    print(featured_data.head())
    print(featured_data.info())

