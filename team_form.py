# Inside team_form.py

from api_integrate import compute_h2h_and_fatigue
import pandas as pd
import numpy as np

# Define this explicitly in team_form.py
FEATURES = ['home_form', 'away_form', 'form_diff', 'fatigue_diff', 'h2h_home_win_rate']

def calculate_rolling_form(df, window=5):
    """
    Calculates points-per-game over the last N matches for every team, 
    strictly using only matches played prior to the current match date.
    """
    print(f"Calculating rolling historical form (Window: {window} games)...")
    
    # 1. Isolate home and away performances into a single "team performance" timeline
    home_df = df[['match_date', 'home_team', 'home_score', 'away_score']].copy()
    home_df.rename(columns={'home_team': 'team', 'home_score': 'team_score', 'away_score': 'opp_score'}, inplace=True)
    home_df['is_home'] = True

    away_df = df[['match_date', 'away_team', 'away_score', 'home_score']].copy()
    away_df.rename(columns={'away_team': 'team', 'away_score': 'team_score', 'home_score': 'opp_score'}, inplace=True)
    away_df['is_home'] = False

    # Combine and sort chronologically
    stacked = pd.concat([home_df, away_df], ignore_index=True)
    stacked = stacked.sort_values('match_date')

    # 2. Calculate match points (3 for win, 1 for draw, 0 for loss)
    conditions = [
        stacked['team_score'] > stacked['opp_score'],
        stacked['team_score'] == stacked['opp_score']
    ]
    stacked['points'] = np.select(conditions, [3, 1], default=0)

    # 3. Calculate Rolling Form 
    # CRITICAL: .shift(1) ensures the result of the CURRENT game isn't included in its own prediction
    stacked['rolling_form'] = stacked.groupby('team')['points'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    
    # Normalize to a 0.0 - 1.0 scale (since max points per game is 3)
    stacked['rolling_form'] = stacked['rolling_form'] / 3.0
    
    # First game of the season? Give them a neutral 0.5 rating
    stacked['rolling_form'] = stacked['rolling_form'].fillna(0.5)

    # 4. Map the calculated forms back to the original DataFrame
    # We use match_date + team as a unique identifier to map correctly
    home_form_map = stacked[stacked['is_home'] == True].set_index(['match_date', 'team'])['rolling_form']
    away_form_map = stacked[stacked['is_home'] == False].set_index(['match_date', 'team'])['rolling_form']

    # Temporarily set index to map the home form
    df = df.set_index(['match_date', 'home_team'])
    df['home_form'] = home_form_map
    df = df.reset_index()

    # Temporarily set index to map the away form
    df = df.set_index(['match_date', 'away_team'])
    df['away_form'] = away_form_map
    df = df.reset_index()

    # Restore perfect chronological order
    df = df.sort_values('match_date').reset_index(drop=True)
    
    # Calculate the differential
    df['form_diff'] = df['home_form'] - df['away_form']

    return df





def engineer_features(dataframe):
    """
    The master feature pipeline. Both history and future matches must pass through here.
    """
    print("Executing master feature engineering pipeline...")
    df = dataframe.copy()
    
    df['match_date'] = pd.to_datetime(df['match_date'])
    df = df.sort_values('match_date').reset_index(drop=True)
    
    # 1. Generate Form
    df = calculate_rolling_form(df, window=5)
    
    # 2. Generate H2H and Fatigue
    df = compute_h2h_and_fatigue(df)
    
    # --- THE X-RAY DEBUGGER ---
    print(f"\n[X-RAY] Total matches BEFORE dropna: {len(df)}")
    print("[X-RAY] Missing values in each column:")
    print(df[FEATURES].isna().sum())
    print("--------------------------------------------------\n")
    # --------------------------
    
    # 3. Filter down to clean data for the model
    df = df.dropna(subset=FEATURES)
    
    print(f"[X-RAY] Total matches AFTER dropna: {len(df)}\n")
    
    return df