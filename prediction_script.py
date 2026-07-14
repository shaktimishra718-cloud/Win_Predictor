import os
import time
import pandas as pd
from dotenv import load_dotenv
from database import get_engine

# ---------------------------------------------------------
# LOCAL PROJECT IMPORTS
# Update these names if your actual functions are named differently
# ---------------------------------------------------------
from api_integrate import fetch_matches, get_real_standings, TARGET_COMPETITIONS, SLEEP_TIME
from team_form import compute_h2h_and_fatigue, FEATURES
# Assuming you save/load your trained model, or import it from execution.py
import joblib

# Load the trained model from the hard drive
try:
    model_temporal = joblib.load('model_temporal.joblib')
except FileNotFoundError:
    print("Error: Model file not found. Please run execution.py first.")
    exit() 

# Load environment variables (API keys, etc.)
load_dotenv()

def load_historical_data():
    """Pulls the entire match history from PostgreSQL into a DataFrame."""
    print("Loading historical match data from PostgreSQL...")
    engine = get_engine()
    
    if not engine:
        raise ConnectionError("Database connection failed. Cannot load history.")

    # Fetch all matches and order them chronologically
    query = "SELECT * FROM matches ORDER BY match_date ASC;"
    df = pd.read_sql(query, engine)
    
    # Ensure dates are parsed correctly
    df['match_date'] = pd.to_datetime(df['match_date'])
    return df

def predict_future_matches():
    """Fetches upcoming fixtures, merges them with DB history, and predicts outcomes."""
    print("--- Phase 4: Fetching Upcoming Fixtures ---")
    upcoming_matches = []

    for comp_name, config in TARGET_COMPETITIONS.items():
        current_season = max(config["seasons"])
        previous_season = current_season - 1

        print(f"Fetching SCHEDULED matches for {comp_name} (Season {current_season})...")

        # Fetch future matches from the API
        df = fetch_matches(config["code"], season=current_season, upcoming_only=True)

        if not df.empty:
            # Fetch current standings to calculate base form
            standings_map = get_real_standings(config["code"], season=current_season)

            # Cross-Season Warm-Up: If the season just started, use last season's form
            is_brand_new_season = len(standings_map) == 0 or all(v.get('form') == 0.5 or not v.get('form') for v in standings_map.values())

            if is_brand_new_season:
                print(f"    Season {current_season} has no history yet. Priming form features with final Season {previous_season} data...")
                standings_map = get_real_standings(config["code"], season=previous_season)

            # Map configuration data
            df['competition'] = comp_name
            df['season'] = current_season
            df['is_neutral_venue'] = int(config["neutral_venues"])

            # Map current standings data to the future fixtures
            df['home_rank'] = df['home_team'].apply(lambda x: standings_map.get(x, {}).get('rank', 10))
            df['away_rank'] = df['away_team'].apply(lambda x: standings_map.get(x, {}).get('rank', 10))
            df['home_form'] = df['home_team'].apply(lambda x: standings_map.get(x, {}).get('form', 0.5))
            df['away_form'] = df['away_team'].apply(lambda x: standings_map.get(x, {}).get('form', 0.5))
            df['form_diff'] = df['home_form'] - df['away_form']

            upcoming_matches.append(df)

        # Respect API rate limits
        time.sleep(SLEEP_TIME)

    if upcoming_matches:
        upcoming_df = pd.concat(upcoming_matches, ignore_index=True)

        print("\nConnecting future fixtures to PostgreSQL historical timeline...")
        
        # Pull the master history from the local database
        master_history = load_historical_data()

        # Flag historical vs future data so we can separate them after engineering
        master_history['is_future'] = False
        upcoming_df['is_future'] = True

        # Combine, sort chronologically, and calculate advanced features
        combined_timeline = pd.concat([master_history, upcoming_df], ignore_index=True)
        combined_timeline['match_date'] = pd.to_datetime(combined_timeline['match_date'], utc=True)
        combined_timeline = combined_timeline.sort_values('match_date').reset_index(drop=True)

        # Run the feature engineering logic (calculates H2H, fatigue, etc.)
        combined_timeline = compute_h2h_and_fatigue(combined_timeline, h2h_n=5, default_rest_days=7)

        # Extract only the future matches back out for inference
        inference_data = combined_timeline[combined_timeline['is_future'] == True].copy()

        # Drop rows where we couldn't calculate essential features
        inference_data = inference_data.dropna(subset=FEATURES)

        if not inference_data.empty:
            print(f"Successfully engineered features for {len(inference_data)} upcoming matches!\n")

            # Isolate the features required by the model
            X_infer = inference_data[FEATURES]

            # Execute the prediction
            probabilities = model_temporal.predict_proba(X_infer)

            inference_data['prob_away_draw'] = probabilities[:, 0]
            inference_data['prob_home_win'] = probabilities[:, 1]

            # --- DASHBOARD OUTPUT ---
            print("=========================================================================")
            print("                 UPCOMING MATCH PREDICTIONS")
            print("=========================================================================")

            # Sort by date so the closest games print first
            inference_data = inference_data.sort_values('match_date')

            for idx, row in inference_data.iterrows():
                home = row['home_team']
                away = row['away_team']
                date = row['match_date'].strftime('%A, %b %d at %H:%M')
                comp = row['competition']

                p_home = row['prob_home_win'] * 100
                p_away_draw = row['prob_away_draw'] * 100

                # Determine the prediction and format the output block
                if p_home > 50:
                    prediction = f"HOME WIN ({home})"
                    confidence = p_home
                    symbol = "[STRONG]" if p_home > 65 else "[MODERATE]"
                else:
                    prediction = f"AWAY/DRAW ({away} avoids defeat)"
                    confidence = p_away_draw
                    symbol = "[STRONG]" if p_away_draw > 65 else "[MODERATE]"

                print(f"[{comp}] {date}")
                print(f"{home} vs {away}")
                print(f"{symbol} Prediction: {prediction} (Confidence: {confidence:.1f}%)")
                print(f"   -> Form Diff: {row['form_diff']:.2f} | Rest Diff: {row['fatigue_diff']} days | H2H Home Win Rate: {row['h2h_home_win_rate']*100:.0f}%")
                print("-" * 73)

        else:
            print("Could not generate features for upcoming matches (data might be incomplete).")
    else:
        print("No upcoming SCHEDULED matches found. (Is it the off-season or an international break?)")

if __name__ == "__main__":
    predict_future_matches()