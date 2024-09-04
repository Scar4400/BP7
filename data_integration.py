import pandas as pd
from data_fetcher import fetch_all_data
from data_preprocessing import preprocess_data
from feature_engineering import create_features

def integrate_data():
    # Fetch data from all sources
    raw_data = fetch_all_data()

    # Preprocess the raw data
    processed_data = preprocess_data(raw_data)

    # Create features
    featured_data = create_features(processed_data)

    return featured_data

def save_integrated_data(data, filename='integrated_data.csv'):
    data.to_csv(filename, index=False)
    print(f"Integrated data saved to {filename}")

def load_integrated_data(filename='integrated_data.csv'):
    return pd.read_csv(filename)

if __name__ == "__main__":
    # Integrate data
    integrated_data = integrate_data()

    # Save integrated data
    save_integrated_data(integrated_data)

    # Load integrated data (for demonstration purposes)
    loaded_data = load_integrated_data()
    print(loaded_data.head())
    print(loaded_data.info())

