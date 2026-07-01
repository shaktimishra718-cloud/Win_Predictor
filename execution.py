
# 3. EXECUTION

historical_dfs = []
all_standings = {}

print(" Phase 1: Ingesting Data from football-data.org ")
for comp_name, config in TARGET_COMPETITIONS.items():
    print(f"Fetching {comp_name}...")
    # Get historical data
    df = fetch_matches(config["code"], upcoming_only=False)
    df['competition'] = comp_name
    df['is_neutral_venue'] = int(config["neutral_venues"])
    historical_dfs.append(df)
    
    # Get real standings and form
    all_standings.update(get_real_standings(config["code"]))
    time.sleep(1) # Pacing

master_history = pd.concat(historical_dfs, ignore_index=True)

# Apply engineering using real API data
master_history['home_rank'] = master_history['home_team'].apply(lambda x: all_standings.get(x, {}).get('rank', 10))
master_history['home_form'] = master_history['home_team'].apply(lambda x: all_standings.get(x, {}).get('form', 0.5))
master_history['home_win'] = (master_history['home_goals'] > master_history['away_goals']).astype(float)


# 4. TRAINING & PREDICTION (Simplified)

FEATURES = ['home_form', 'home_rank', 'is_neutral_venue']
X_train = master_history[FEATURES].fillna(0.5)
y_train = master_history['home_win']

model = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
print("Pipeline trained on real football-data.org standings!")
