historical_dfs = []
seasons = [2023, 2024, 2025] 
REQUESTS_PER_MINUTE = 10
SLEEP_TIME = (60 / REQUESTS_PER_MINUTE) + 1

print("--- Phase 3: Processing Matches with Multi-Season Historical Data ---")

for comp_name, config in TARGET_COMPETITIONS.items():
    for season in config["seasons"]:
        print(f"Processing {comp_name} for Season {season}...")
        df = fetch_matches(config["code"], season=season, upcoming_only=False)
        
        if not df.empty:
            standings_map = get_real_standings(config["code"], season=season)
            
            df['competition'] = comp_name
            df['season'] = season
            df['is_neutral_venue'] = int(config["neutral_venues"])
            
            df['home_rank'] = df['home_team'].apply(lambda x: standings_map.get(x, {}).get('rank', 10))
            df['away_rank'] = df['away_team'].apply(lambda x: standings_map.get(x, {}).get('rank', 10))
            df['home_form'] = df['home_team'].apply(lambda x: standings_map.get(x, {}).get('form', 0.5))
            df['away_form'] = df['away_team'].apply(lambda x: standings_map.get(x, {}).get('form', 0.5))
            df['form_diff'] = df['home_form'] - df['away_form']
            
            df['home_win'] = (df['home_goals'] > df['away_goals']).astype(float)
            historical_dfs.append(df)
            
        time.sleep(SLEEP_TIME)

# Combine and Sort Chronologically (CRITICAL)
master_history = pd.concat(historical_dfs, ignore_index=True)
master_history['match_date'] = pd.to_datetime(master_history['match_date'])
master_history = master_history.sort_values('match_date').reset_index(drop=True)

# Apply your custom historical feature function
print("Calculating historical features (H2H, Fatigue)...")
master_history = compute_h2h_and_fatigue(master_history, h2h_n=5, default_rest_days=7)

print(f"\nPipeline initialized with {len(master_history)} total historical matches.")

# --- 4. ADVANCED TEMPORAL TRAINING & VALIDATION ---



FEATURES = [
    'home_rank', 
    'away_rank', 
    'form_diff', 
    'is_neutral_venue',
    # Your advanced H2H features
    'h2h_home_win_rate', 
    'h2h_draw_rate', 
    'h2h_away_win_rate',
    'h2h_home_avg_goals_for',
    'h2h_home_avg_goals_against',
    # Your advanced Fatigue features
    'days_since_home_last',
    'days_since_away_last',
    'fatigue_diff'         
]

# Drop rows with missing historical data to prevent model crash
master_history = master_history.dropna(subset=FEATURES + ['home_win'])

X = master_history[FEATURES]
y = master_history['home_win']

# =====================================================================
# APPROACH 1: STRICT TEMPORAL SPLIT (80% Past / 20% Future)
# =====================================================================
print("\n--- APPROACH 1: STRICT TEMPORAL VALIDATION ---")
split_idx = int(len(master_history) * 0.8)

X_train_temp, X_test_temp = X.iloc[:split_idx], X.iloc[split_idx:]
y_train_temp, y_test_temp = y.iloc[:split_idx], y.iloc[split_idx:]

model_temporal = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_temporal.fit(X_train_temp, y_train_temp)

y_pred_test_temp = model_temporal.predict(X_test_temp)
test_accuracy = accuracy_score(y_test_temp, y_pred_test_temp)

print(f"Training set: {len(X_train_temp)} samples | Test set: {len(X_test_temp)} samples")
print(f"Test Accuracy (Unseen Future Data): {test_accuracy*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test_temp, y_pred_test_temp))

# =====================================================================
# APPROACH 2: TIME-SERIES CROSS VALIDATION
# =====================================================================
# ✅ FIX 2: Replaced random K-Fold with TimeSeriesSplit to prevent leakage
print("\n--- APPROACH 2: TIME-SERIES CROSS-VALIDATION ---")
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for train_index, test_index in tscv.split(X):
    X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
    
    model_cv = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model_cv.fit(X_train_cv, y_train_cv)
    cv_scores.append(accuracy_score(y_test_cv, model_cv.predict(X_test_cv)))

print(f"Chronological Fold Scores: {[f'{s*100:.2f}%' for s in cv_scores]}")
print(f"Mean Time-Series CV Accuracy: {np.mean(cv_scores)*100:.2f}%")
print(f"Std Dev: {np.std(cv_scores)*100:.2f}%")

# =====================================================================
# FEATURE IMPORTANCE
# =====================================================================
print("\n--- FEATURE IMPORTANCE ---")
importances = pd.DataFrame({'feature': FEATURES, 'importance': model_temporal.feature_importances_})
print(importances.sort_values(by='importance', ascending=False).to_string(index=False))

# =====================================================================
# SUMMARY
# =====================================================================
print("\n--- MODEL VALIDATION SUMMARY ---")
print(f"Temporal Test Accuracy (20% holdout): {test_accuracy*100:.2f}%")
print(f"Time-Series CV Mean Accuracy:         {np.mean(cv_scores)*100:.2f}%")

if test_accuracy < 0.55:
    print("\n⚠️ WARNING: Temporal test accuracy is low. Model may not generalize well.")
else:
    print("\n✓ Temporal test accuracy is stable. Advanced features successfully integrated.")
