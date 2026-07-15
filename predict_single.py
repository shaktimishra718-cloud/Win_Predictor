import pandas as pd
import joblib
from datetime import datetime
from database import get_engine
from team_form import engineer_features

# The exact features your model requires
FEATURES = ['home_form', 'away_form', 'form_diff', 'fatigue_diff', 'h2h_home_win_rate']

def get_latest_team_stats(df, team_name):
    """Finds the most recent match a team played to get their current form."""
    team_matches = df[(df['home_team'] == team_name) | (df['away_team'] == team_name)]
    if team_matches.empty:
        return None
    
    # Get their absolute latest match
    latest_match = team_matches.iloc[-1]
    
    # Extract their form from that match depending on if they were home or away
    if latest_match['home_team'] == team_name:
        form = latest_match['home_form']
    else:
        form = latest_match['away_form']
        
    return {
        'form': form,
        'last_match_date': latest_match['match_date']
    }

def calculate_h2h(df, home_team, away_team, n=5):
    """Calculates the recent Head-to-Head win rate for the Home Team."""
    h2h_matches = df[
        ((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
        ((df['home_team'] == away_team) & (df['away_team'] == home_team))
    ].tail(n)
    
    if h2h_matches.empty:
        return 0.5 # Default to 50/50 if they have never played
        
    home_wins = 0
    for _, match in h2h_matches.iterrows():
        if match['home_team'] == home_team and match['home_score'] > match['away_score']:
            home_wins += 1
        elif match['away_team'] == home_team and match['away_score'] > match['home_score']:
            home_wins += 1
            
    return home_wins / len(h2h_matches)

def run_interactive_predictor():
    print("\n" + "="*40)
    print(" 🔮 LIVE MATCH PREDICTOR 🔮 ")
    print("="*40)
    
    home_team = input("Enter the Home Team (e.g., Arsenal): ").strip()
    away_team = input("Enter the Away Team (e.g., Chelsea): ").strip()
    
    print("\nLoading database and calculating current form...")
    
    # 1. Load data and model
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM matches ORDER BY match_date ASC", engine)
    
    # Convert dates to UTC to avoid timezone crashes
    df['match_date'] = pd.to_datetime(df['match_date'], utc=True)
    
    # Calculate historical rolling features
    df = engineer_features(df)
    
    try:
        model = joblib.load('model_temporal.joblib')
    except FileNotFoundError:
        print("Error: Could not find 'model_temporal.joblib'. Please run execution.py first.")
        return

    # 2. Extract current stats
    home_stats = get_latest_team_stats(df, home_team)
    away_stats = get_latest_team_stats(df, away_team)
    
    if not home_stats:
        print(f"Error: Could not find any historical data for '{home_team}'. Check spelling.")
        return
    if not away_stats:
        print(f"Error: Could not find any historical data for '{away_team}'. Check spelling.")
        return

    # 3. Calculate math for TODAY
    today = pd.Timestamp.utcnow()
    home_rest = (today - home_stats['last_match_date']).days
    away_rest = (today - away_stats['last_match_date']).days
    
    h2h_rate = calculate_h2h(df, home_team, away_team)
    
    # 4. Build the final feature row for the model
    match_features = pd.DataFrame([{
        'home_form': home_stats['form'],
        'away_form': away_stats['form'],
        'form_diff': home_stats['form'] - away_stats['form'],
        'fatigue_diff': home_rest - away_rest,
        'h2h_home_win_rate': h2h_rate
    }])

    # 5. Predict with Probabilities
    # predict_proba returns a 2D array: [[Probability of 0, Probability of 1]]
    probabilities = model.predict_proba(match_features)[0]
    draw_away_prob = probabilities[0] * 100
    home_win_prob = probabilities[1] * 100
    
    print("\n" + "-"*40)
    print(f" MATCHUP: {home_team} (Home) vs {away_team} (Away)")
    print("-"*40)
    print(f" {home_team} Current Form: {home_stats['form']:.2f}")
    print(f" {away_team} Current Form: {away_stats['form']:.2f}")
    print(f" H2H Win Rate ({home_team}): {h2h_rate * 100:.1f}%")
    print("\n 📊 PREDICTION RESULTS:")
    print(f" Probability of {home_team} Win:      {home_win_prob:.1f}%")
    print(f" Probability of Draw/Away Win:  {draw_away_prob:.1f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_interactive_predictor()