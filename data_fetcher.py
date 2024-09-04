import requests
import pandas as pd
from config import PINNACLE_URL, LIVESCORE_URL, API_FOOTBALL_URL, HEADERS

def fetch_pinnacle_data():
    querystring = {"is_have_odds": "true", "sport_id": "1"}
    response = requests.get(PINNACLE_URL, headers=HEADERS["pinnacle"], params=querystring)
    return response.json()

def fetch_livescore_data():
    querystring = {"Category": "soccer", "Query": ""}
    response = requests.get(LIVESCORE_URL, headers=HEADERS["livescore"], params=querystring)
    return response.json()

def fetch_api_football_data():
    querystring = {"page": "1"}
    response = requests.get(API_FOOTBALL_URL, headers=HEADERS["api_football"], params=querystring)
    return response.json()

def fetch_all_data():
    pinnacle_data = fetch_pinnacle_data()
    livescore_data = fetch_livescore_data()
    api_football_data = fetch_api_football_data()

    return {
        "pinnacle": pinnacle_data,
        "livescore": livescore_data,
        "api_football": api_football_data
    }

if __name__ == "__main__":
    data = fetch_all_data()
    print("Data fetched successfully")
    print(f"Pinnacle data: {len(data['pinnacle'])} items")
    print(f"Livescore data: {len(data['livescore'])} items")
    print(f"API Football data: {len(data['api_football'])} items")

