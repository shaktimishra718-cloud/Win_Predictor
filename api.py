import pandas as pd
import joblib
from datetime import datetime
from enum import Enum
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from database import get_engine
from team_form import engineer_features

# Initialize the FastAPI application
app = FastAPI(
    title="Win_Predictor Match Forecasting API",
    description="A production-ready REST API serving Random Forest predictions with dynamic database routing.",
    version="1.1.0"
)

# Define the features the model expects
FEATURES = ['home_form', 'away_form', 'form_diff', 'fatigue_diff', 'h2h_home_win_rate']

# --- The Dropdown Menu Options ---
class CompetitionOption(str, Enum):
    premier_league = "Premier League"
    la_liga = "La Liga"
    champions_league = "Champions League"

# Pydantic schema for structured API responses
class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    home_form: float
    away_form: float
    h2h_home_win_rate: float
    home_win_probability: float
    draw_away_probability: float
    status: str

def get_latest_team_stats(df, team_name):
    """Scans the calculated historical dataframe to find a team's most recent form data."""
    team_matches = df[(df['home_team'] == team_name) | (df['away_team'] == team_name)]
    if team_matches.empty:
        return None
    
    latest_match = team_matches.iloc[-1]
    
    if latest_match['home_team'] == team_name:
        form = latest_match['home_form']
    else:
        form = latest_match['away_form']
        
    return {
        'form': form,
        'last_match_date': latest_match['match_date']
    }

def calculate_h2h(df, home_team, away_team, n=5):
    """Computes historical Head-to-Head performance between the two teams."""
    h2h_matches = df[
        ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
        ((df['home_team'] == away_team) & (df['away_team'] == home_team))
    ].tail(n)
    
    if h2h_matches.empty:
        return 0.5
        
    home_wins = 0
    for _, match in h2h_matches.iterrows():
        if match['home_team'] == home_team and match['home_score'] > match['away_score']:
            home_wins += 1
        elif match['away_team'] == home_team and match['away_score'] > match['home_score']:
            home_wins += 1
            
    return home_wins / len(h2h_matches)


@app.get("/predict", response_model=PredictionResponse)
def predict_match(
    competition: CompetitionOption = Query(..., description="Select the specific competition to route the database query"),
    home_team: str = Query(..., description="Name of the Home Team (e.g., Real Madrid)", example="Real Madrid"),
    away_team: str = Query(..., description="Name of the Away Team (e.g., FC Barcelona)", example="FC Barcelona")
):
    """
    Dynamically routes the request to an isolated PostgreSQL table, calculates rolling 
    statistics, and returns the probability breakdown using the Random Forest pipeline.
    """
    home_team = home_team.strip()
    away_team = away_team.strip()

    # 1. Establish database connection
    engine = get_engine()
    if not engine:
        raise HTTPException(status_code=500, detail="Database connection failed.")
        
    try:
        # THE ROUTER: Map the user's dropdown choice to the correct physical table
        table_map = {
            "Premier League": "matches_pl",
            "La Liga": "matches_pd",
            "Champions League": "matches_cl"
        }
        target_table = table_map[competition.value]
        
        # Dynamically query only the isolated table
        query = f"SELECT * FROM {target_table} ORDER BY match_date ASC"
        df = pd.read_sql(query, engine)
        
    except Exception as e:
        # If the table doesn't exist yet, it will throw an error here
        raise HTTPException(status_code=500, detail=f"Failed to query database table '{target_table}': {str(e)}")
        
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No match data found in table {target_table}.")

    # 2. Process dates and engineer rolling features
    df['match_date'] = pd.to_datetime(df['match_date'], utc=True)
    df = engineer_features(df)
    
    # 3. Load the saved machine learning model
    try:
        model = joblib.load('model_temporal.joblib')
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Prediction model 'model_temporal.joblib' not found. Please train the model first.")

    # 4. Extract recent stats for both inputs
    home_stats = get_latest_team_stats(df, home_team)
    away_stats = get_latest_team_stats(df, away_team)
    
    if not home_stats:
        raise HTTPException(status_code=400, detail=f"Could not find {competition.value} records for team: '{home_team}'.")
    if not away_stats:
        raise HTTPException(status_code=400, detail=f"Could not find {competition.value} records for team: '{away_team}'.")

    # 5. Calculate real-time fatigue and H2H statistics
    today = pd.Timestamp.utcnow()
    home_rest = (today - home_stats['last_match_date']).days
    away_rest = (today - away_stats['last_match_date']).days
    h2h_rate = calculate_h2h(df, home_team, away_team)
    
    # 6. Build the input structure for model inference
    match_features = pd.DataFrame([{
        'home_form': home_stats['form'],
        'away_form': away_stats['form'],
        'form_diff': home_stats['form'] - away_stats['form'],
        'fatigue_diff': home_rest - away_rest,
        'h2h_home_win_rate': h2h_rate
    }])

    # 7. Generate confidence probabilities
    probabilities = model.predict_proba(match_features)[0]
    draw_away_prob = float(probabilities[0] * 100)
    home_win_prob = float(probabilities[1] * 100)
    
    # Return response structured by the Pydantic schema
    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_form": round(home_stats['form'], 2),
        "away_form": round(away_stats['form'], 2),
        "h2h_home_win_rate": round(h2h_rate * 100, 1),
        "home_win_probability": round(home_win_prob, 1),
        "draw_away_probability": round(draw_away_prob, 1),
        "status": "Success"
    }

@app.get("/")
def read_root():
    return {"message": "Win_Predictor API is fully operational with Multi-Table Routing. Navigate to /docs for the interactive interface."}