import os
from dotenv import load_dotenv

load_dotenv()

PINNACLE_API_KEY = os.getenv("PINNACLE_API_KEY")
LIVESCORE_API_KEY = os.getenv("LIVESCORE_API_KEY")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")

PINNACLE_URL = "<https://pinnacle-odds.p.rapidapi.com/kit/v1/special-markets>"
LIVESCORE_URL = "<https://livescore6.p.rapidapi.com/v2/search>"
API_FOOTBALL_URL = "<https://api-football-v1.p.rapidapi.com/v2/odds/league/865927/bookmaker/5>"

HEADERS = {
    "pinnacle": {
        "x-rapidapi-key": PINNACLE_API_KEY,
        "x-rapidapi-host": "pinnacle-odds.p.rapidapi.com"
    },
    "livescore": {
        "x-rapidapi-key": LIVESCORE_API_KEY,
        "x-rapidapi-host": "livescore6.p.rapidapi.com"
    },
    "api_football": {
        "x-rapidapi-key": API_FOOTBALL_KEY,
        "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
    }
}

