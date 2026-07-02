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


#added a labelling
def label_match(row):
   #labelling
    if row['home_goals'] > row['away_goals']:
        return 2  # Home Win
    elif row['home_goals'] == row['away_goals']:
        return 1  # Draw
    else:
        return 0  # Away Win

def fetch_matches(code, season, upcoming_only=False):
    """Fetches matches for a specific season using the retry logic."""
    status = "SCHEDULED" if upcoming_only else "FINISHED"
    url = f"{BASE_URL}/competitions/{code}/matches?status={status}&season={season}"
    
    # Using the new retry helper
    response = fetch_with_retry(url, HEADERS)
    
    # Check if the response is valid
    if response is None or response.status_code != 200:
        return pd.DataFrame()
        
    data = response.json()
    matches = []
    
    for item in data.get("matches", []):
        matches.append({
            "match_date": item["utcDate"],
            "home_team": item["homeTeam"]["name"],
            "away_team": item["awayTeam"]["name"],
            "home_goals": item["score"]["fullTime"]["home"] if item["score"]["fullTime"]["home"] is not None else 0,
            "away_goals": item["score"]["fullTime"]["away"] if item["score"]["fullTime"]["away"] is not None else 0
        })
    df = pd.DataFrame(matches)

    if not df.empty:
        df['match_result'] = df.apply(label_match, axis=1)
        
    return df


def get_real_standings(code, season=2025):
    """Fetches standings for a specific season."""
    # Update the URL to include the season parameter
    data = get_data(f"/competitions/{code}/standings?season={season}")
    standings_map = {}
    
    if 'standings' in data:
        for table in data['standings']:
            if table['type'] == 'TOTAL':
                for pos in table['table']:
                    form_str = pos.get('form', '') or ""
                    # Ensure you handle the case where form_str is empty
                    scores = [3 if c == 'W' else 1 if c == 'D' else 0 for c in form_str.split(',')]
                    # Use length of scores to avoid division by zero if no form data
                    score = sum(scores) / len(scores) if scores else 0.5
                    standings_map[pos['team']['name']] = {'rank': pos['position'], 'form': score}
    return standings_map
