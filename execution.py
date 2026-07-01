# THE MAIN EXECUTION
print(" Phase 1: Fetching & Engineering Historical Data ")
historical_dfs = []

for comp_name, config in TARGET_COMPETITIONS.items():
    df = fetch_competition_data(API_KEY, config["id"], comp_name, config["neutral_venues"], SEASON, upcoming_only=False)
    if not df.empty:
        historical_dfs.append(df)
    time.sleep(7) # Safe API pacing

master_history = pd.concat(historical_dfs, ignore_index=True)

# Generate mock standings based on the teams we just downloaded
all_teams = master_history['home_team'].tolist() + master_history['away_team'].tolist()
standings_map = get_mock_standings(all_teams)

# Apply our new Feature Engineering math
master_history = feature_engineering(master_history, standings_map)

print("\n Phase 2: Training the Context-Aware Model ")
# WE DELETED THE ARBITRARY TEAM CODES! The model only looks at performance now.
FEATURES = ['home_form', 'away_form', 'home_rank', 'away_rank', 'is_neutral_venue']


X_train = master_history[FEATURES]
y_train = master_history['home_win']

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
print("Model successfully trained on form and standings.")


print("\n Phase 3: Predicting Upcoming Fixtures ")
for comp_name, config in TARGET_COMPETITIONS.items():
    future_df = fetch_competition_data(API_KEY, config["id"], comp_name, config["neutral_venues"], SEASON, upcoming_only=True)
    
    if future_df.empty:
        continue

print(f"\n============= {comp_name.upper()} PREDICTIONS =============")
    
for idx, match in future_df.iterrows():
     home_t = match['home_team']
     away_t = match['away_team']
     
     # Look up the latest known form for these teams from our historical dataset
     try:
            current_home_form = master_history[master_history['home_team'] == home_t]['home_form'].iloc[-1]
     except IndexError:
            current_home_form = 0.5 # Default if brand new team
            
     try:
            current_away_form = master_history[master_history['away_team'] == away_t]['away_form'].iloc[-1]
     except IndexError:
            current_away_form = 0.5
            
     h_rank = standings_map.get(home_t, 10)
     a_rank = standings_map.get(away_t, 10)
        
     # Package the real-time stats into a format the model can read
     match_features = pd.DataFrame([{
            'home_form': current_home_form,
            'away_form': current_away_form,
            'home_rank': h_rank,
            'away_rank': a_rank,
            'is_neutral_venue': int(config["neutral_venues"])
        }])
        
     # Predict!
     prob = model.predict_proba(match_features)[0][1] * 100
        
     print(f"{home_t} (Form: {current_home_form:.2f}, Rank: {h_rank})")
      
     print(f"vs {away_t} (Form: {current_away_form:.2f}, Rank: {a_rank})")
     print(f"-> AI Predicted Home Win: {prob:.1f}%\n")
        
     time.sleep(7)
