import os
from dotenv import load_dotenv

# Load the hidden variables
load_dotenv()
import requests
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier

from time_limit import fetch_with_retry


# 1. CONFIGURATION

# Mapping codes used by football-data.org
TARGET_COMPETITIONS = {
    "Champions League": {
        "code": "CL",
        "neutral_venues": True, # UCL Finals are often on neutral ground
        "seasons": [2023, 2024, 2025,2026]
    },
    "Premier League": {
        "code": "PL",
        "neutral_venues": False,
        "seasons": [2023, 2024, 2025,2026]
    },
    "La Liga": {
        "code": "PD",
        "neutral_venues": False,
        "seasons": [2023, 2024, 2025,2026]
    }
}


api_key = os.getenv('FOOTBALL_API_KEY')
HEADERS = {'X-Auth-Token': api_key}
BASE_URL = "https://api.football-data.org/v4"
REQUESTS_PER_MINUTE = 10
SLEEP_TIME = 7

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
def compute_h2h_and_fatigue(df, h2h_n=5, default_rest_days=7):
    """
    Adds H2H and fatigue features to df in place and returns df.
    h2h_n: number of previous encounters to use for H2H stats (most recent up to h2h_n).
    default_rest_days: days to use when a team has no previous match in history.
    """
    # Prepare new columns with defaults
    df['h2h_home_win_rate'] = 0.5
    df['h2h_draw_rate'] = 0.0
    df['h2h_away_win_rate'] = 0.5
    df['h2h_home_avg_goals_for'] = 0.0
    df['h2h_home_avg_goals_against'] = 0.0
    df['days_since_home_last'] = default_rest_days
    df['days_since_away_last'] = default_rest_days
    df['fatigue_diff'] = 0.0  # home_rest - away_rest

    # In-memory state
    pair_history = {}   # key: tuple(sorted([teamA, teamB])) -> list of past matches (dict)
    last_match_date = {}  # key: team -> last match datetime

    # Iterate chronologically
    for idx, row in df.iterrows():
        date = row['match_date']
        home = row['home_team']
        away = row['away_team']
        key = tuple(sorted([home, away]))

        # ---- H2H stats from previous encounters ----
        past = pair_history.get(key, [])  # already chronological
        if past:
            last_encounters = past[-h2h_n:]  # last up to h2h_n encounters
            n = len(last_encounters)
            home_wins = 0
            draws = 0
            away_wins = 0
            goals_for = 0
            goals_against = 0
            for m in last_encounters:
                # determine goals for 'home' team in the current row, regardless of original home/away in m
                if m['home_team'] == home:
                    gf = m['home_goals']; ga = m['away_goals']
                else:
                    # roles were flipped in that past match
                    gf = m['away_goals']; ga = m['home_goals']

                goals_for += gf
                goals_against += ga
                if gf > ga:
                    home_wins += 1
                elif gf == ga:
                    draws += 1
                else:
                    away_wins += 1

            # set rates
            df.at[idx, 'h2h_home_win_rate'] = home_wins / n
            df.at[idx, 'h2h_draw_rate'] = draws / n
            df.at[idx, 'h2h_away_win_rate'] = away_wins / n
            df.at[idx, 'h2h_home_avg_goals_for'] = goals_for / n
            df.at[idx, 'h2h_home_avg_goals_against'] = goals_against / n
        else:
            # no history: keep sensible defaults (0.5 win-rate is neutral)
            df.at[idx, 'h2h_home_win_rate'] = 0.5
            df.at[idx, 'h2h_draw_rate'] = 0.0
            df.at[idx, 'h2h_away_win_rate'] = 0.5
            df.at[idx, 'h2h_home_avg_goals_for'] = 0.0
            df.at[idx, 'h2h_home_avg_goals_against'] = 0.0

        # ---- Fatigue: days since last match for each team ----
        if home in last_match_date:
            days_home = (date - last_match_date[home]).days
        else:
            days_home = default_rest_days

        if away in last_match_date:
            days_away = (date - last_match_date[away]).days
        else:
            days_away = default_rest_days

        # 🚨 THE FIX: Apply the Biological Cap (Max 14 days)
        days_home = min(days_home, 14)
        days_away = min(days_away, 14)

        df.at[idx, 'days_since_home_last'] = days_home
        df.at[idx, 'days_since_away_last'] = days_away
        df.at[idx, 'fatigue_diff'] = days_home - days_away

        # ---- Update state with the current match so it will be seen by later matches ----
        # store only the fields we need to compute H2H later
        h_score = row.get('home_score', row.get('home_goals', 0))
        a_score = row.get('away_score', row.get('away_goals', 0))
        
        # 2. Safely handle NaN (future matches) by converting them to 0
        h_score = 0 if pd.isna(h_score) else int(h_score)
        a_score = 0 if pd.isna(a_score) else int(a_score)
        match_record = {
            'match_date': date,
            'home_team': home,
            'away_team': away,
            'home_goals': h_score,
            'away_goals': a_score,
        }
        pair_history.setdefault(key, []).append(match_record)
        last_match_date[home] = date
        last_match_date[away] = date

        if 'h2h_home_win_rate' in df.columns:
            df['h2h_home_win_rate'] = df['h2h_home_win_rate'].fillna(0.5)
        
        # 2. Fill missing Fatigue data (Assume 0 difference in rest if it's game 1)
        if 'fatigue_diff' in df.columns:
            df['fatigue_diff'] = df['fatigue_diff'].fillna(0)
    return df


