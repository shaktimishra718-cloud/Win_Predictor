from google.colab import userdata
import requests
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier


# CONFIGURATION

TARGET_COMPETITIONS = {
    "World Cup": {"id": "1", "neutral_venues": True},
    "Premier League": {"id": "39", "neutral_venues": False},
    "La Liga": {"id": "140", "neutral_venues": False}
}

SEASON = "2026"
API_KEY = userdata.get('APISPORTS_KEY')


# DATA INGESTION FUNCTIONS
def fetch_competition_data(api_key, league_id, comp_name, is_neutral, season, upcoming_only):
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": api_key}
    
    if upcoming_only:
        querystring = {"league": league_id, "season": season, "next": "10"}
    else:
        querystring = {"league": league_id, "season": season, "last": "60"} 
        
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code != 200: return pd.DataFrame()
        
    matches = []
    for item in response.json().get("response", []):
        matches.append({
            "fixture_id": item["fixture"]["id"],
            "match_date": item["fixture"]["date"],
            "competition": comp_name,
            "home_team": item["teams"]["home"]["name"],
            "away_team": item["teams"]["away"]["name"],
            "is_neutral_venue": int(is_neutral),
            "home_goals": item["goals"]["home"],
            "away_goals": item["goals"]["away"]
        })
    return pd.DataFrame(matches)

def get_mock_standings(teams):
    
    unique_teams = list(set(teams))
    # Randomly assign a rank to each team to simulate last year's table
    ranks = np.random.permutation(len(unique_teams)) + 1 
    return dict(zip(unique_teams, ranks))


