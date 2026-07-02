# 3. EXECUTION WITH MULTI-SEASON LOOP
historical_dfs = []
seasons = [2023, 2024, 2025] # Defined seasons for historical analysis
REQUESTS_PER_MINUTE = 10
SLEEP_TIME = (60 / REQUESTS_PER_MINUTE) + 1

print("--- Phase 3: Processing Matches with Multi-Season Historical Data ---")

for comp_name, config in TARGET_COMPETITIONS.items():
    for season in config["seasons"]:
        print(f"Processing {comp_name} for Season {season}...")
      
      
        df = fetch_matches(config["code"], season=season, upcoming_only=False)
        
        if not df.empty:
            
            # for strict historical accuracy season-specific standings
            standings_map = get_real_standings(config["code"])
            
            df['competition'] = comp_name
            df['season'] = season
            df['is_neutral_venue'] = int(config["neutral_venues"])
            
            # 3. Map Features
            df['home_rank'] = df['home_team'].apply(lambda x: standings_map.get(x, {}).get('rank', 10))
            df['away_rank'] = df['away_team'].apply(lambda x: standings_map.get(x, {}).get('rank', 10))
            df['home_form'] = df['home_team'].apply(lambda x: standings_map.get(x, {}).get('form', 0.5))
            df['away_form'] = df['away_team'].apply(lambda x: standings_map.get(x, {}).get('form', 0.5))
            df['form_diff'] = df['home_form'] - df['away_form']
            
            # Target
            df['home_win'] = (df['home_goals'] > df['away_goals']).astype(float)
            historical_dfs.append(df)
        

        time.sleep(SLEEP_TIME)

# Combine all seasons into one master dataframe
master_history = pd.concat(historical_dfs, ignore_index=True)
print(f"Pipeline initialized with {len(master_history)} total historical matches.")

# 4. TRAINING & PREDICTION
FEATURES = ['form_diff', 'home_rank', 'away_rank', 'is_neutral_venue']
X = master_history[FEATURES].fillna(0.5)
y = master_history['home_win']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
