
# 3. EXECUTION

historical_dfs = []
all_standings = {}

print("--- Phase 3: Processing Matches with Form Differential ---")
for comp_name, config in TARGET_COMPETITIONS.items():
    print(f"Processing {comp_name}...")
    
    # Get standings for both Home and Away lookup
    standings_map = get_real_standings(config["code"])
    all_standings.update(standings_map)
    
    # Get matches
    df = fetch_matches(config["code"], upcoming_only=False)
    if not df.empty:
        df['competition'] = comp_name
        df['is_neutral_venue'] = int(config["neutral_venues"])
        
        # ADDED: Map both home AND away form/rank
        df['home_rank'] = df['home_team'].apply(lambda x: standings_map.get(x, {}).get('rank', 10))
        df['away_rank'] = df['away_team'].apply(lambda x: standings_map.get(x, {}).get('rank', 10))
        df['home_form'] = df['home_team'].apply(lambda x: standings_map.get(x, {}).get('form', 0.5))
        df['away_form'] = df['away_team'].apply(lambda x: standings_map.get(x, {}).get('form', 0.5))
        
        # ENGINEERED FEATURE: Form Differential (Higher = Home Team is in better form)
        df['form_diff'] = df['home_form'] - df['away_form']
        
        # Target: Home Win
        df['home_win'] = (df['home_goals'] > df['away_goals']).astype(float)
        historical_dfs.append(df)
    
    time.sleep(1) # API Pacing

master_history = pd.concat(historical_dfs, ignore_index=True)
# 4. TRAINING & PREDICTION (Simplified)

FEATURES = ['form_diff', 'home_rank', 'away_rank', 'is_neutral_venue']

X = master_history[FEATURES].fillna(0.5)
y = master_history['home_win']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
print("Model trained on Form Differential and Relative Ranking!")
