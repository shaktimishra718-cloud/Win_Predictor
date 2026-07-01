from google.colab import userdata
import requests
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier


# 1. CONFIGURATION

# Mapping codes used by football-data.org
TARGET_COMPETITIONS = {
    "World Cup": {"code": "WC", "neutral_venues": True},
    "Premier League": {"code": "PL", "neutral_venues": False},
    "La Liga": {"code": "PD", "neutral_venues": False}
}

TOKEN = userdata.get('footdata_key')
HEADERS = {'X-Auth-Token': TOKEN}
BASE_URL = "https://api.football-data.org/v4"


# 2. DATA INGESTION FUNCTIONS

def get_data(endpoint):
    response = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS)
    return response.json() if response.status_code == 200 else {}

def fetch_matches(code, upcoming_only=False):
    """Fetches matches and extracts relevant fields."""
    status = "SCHEDULED" if upcoming_only else "FINISHED"
    data = get_data(f"/competitions/{code}/matches?status={status}")
    matches = []
    
    for item in data.get("matches", []):
        matches.append({
            "match_date": item["utcDate"],
            "home_team": item["homeTeam"]["name"],
            "away_team": item["awayTeam"]["name"],
            "home_goals": item["score"]["fullTime"]["home"] if item["score"]["fullTime"]["home"] is not None else 0,
            "away_goals": item["score"]["fullTime"]["away"] if item["score"]["fullTime"]["away"] is not None else 0
        })
    return pd.DataFrame(matches)

def get_real_standings(code):
    """Fetches real standings/form from the API."""
    data = get_data(f"/competitions/{code}/standings")
    standings_map = {}
    if 'standings' in data:
        for table in data['standings']:
            if table['type'] == 'TOTAL':
                for pos in table['table']:
                    # Create a score: 3 for W, 1 for D, 0 for L
                    form_str = pos.get('form', '') or ""
                    score = sum([3 if c == 'W' else 1 if c == 'D' else 0 for c in form_str.split(',')]) / 15
                    standings_map[pos['team']['name']] = {'rank': pos['position'], 'form': score}
    return standings_map
