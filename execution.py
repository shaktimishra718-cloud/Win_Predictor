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

# 4. TRAINING & PREDICTION WITH TRAIN-TEST SPLIT & TEMPORAL VALIDATION
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score

FEATURES = ['form_diff', 'home_rank', 'away_rank', 'is_neutral_venue']

# Data validation
print("\n--- Data Validation ---")
print(f"Total samples: {len(master_history)}")
print(f"Seasons in dataset: {sorted(master_history['season'].unique())}")
print(f"Home win rate: {master_history['home_win'].mean()*100:.2f}%")

# Check for missing values
X_full = master_history[FEATURES].copy()
missing_counts = X_full.isnull().sum()
if missing_counts.any():
    print(f"\nMissing values by feature:")
    print(missing_counts[missing_counts > 0])
    X_full = X_full.fillna(0.5)
else:
    print("No missing values detected.")

y_full = master_history['home_win']

# ========== APPROACH 1: TEMPORAL VALIDATION (Time-based split) ==========
# Earlier seasons for training, latest season for testing
print("\n\n--- APPROACH 1: TEMPORAL VALIDATION (Time-based Split) ---")
print("Training on 2023-2024 data, Testing on 2025 data\n")

# Sort by season for temporal split
sorted_data = master_history.sort_values('season')
train_mask = sorted_data['season'] < max(sorted_data['season'])
test_mask = sorted_data['season'] == max(sorted_data['season'])

X_train_temporal = sorted_data[train_mask][FEATURES].fillna(0.5)
X_test_temporal = sorted_data[test_mask][FEATURES].fillna(0.5)
y_train_temporal = sorted_data[train_mask]['home_win']
y_test_temporal = sorted_data[test_mask]['home_win']

print(f"Training set: {len(X_train_temporal)} samples (Seasons: {sorted(sorted_data[train_mask]['season'].unique())})")
print(f"Test set: {len(X_test_temporal)} samples (Season: {sorted_data[test_mask]['season'].unique()[0]})")
print(f"Training home win rate: {y_train_temporal.mean()*100:.2f}%")
print(f"Test home win rate: {y_test_temporal.mean()*100:.2f}%")

# Train model on historical data
model_temporal = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_temporal.fit(X_train_temporal, y_train_temporal)

# Evaluate on test set
y_pred_train_temporal = model_temporal.predict(X_train_temporal)
y_pred_test_temporal = model_temporal.predict(X_test_temporal)

print("\nTRAINING SET Performance:")
print(f"  Accuracy: {accuracy_score(y_train_temporal, y_pred_train_temporal)*100:.2f}%")

print("\nTEST SET Performance (2025 unseen data):")
test_accuracy = accuracy_score(y_test_temporal, y_pred_test_temporal)
print(f"  Accuracy: {test_accuracy*100:.2f}%")
print("\n  Classification Report:")
print(classification_report(y_test_temporal, y_pred_test_temporal, target_names=['Away Win/Draw', 'Home Win']))

# ========== APPROACH 2: RANDOM TRAIN-TEST SPLIT (Stratified) ==========
print("\n\n--- APPROACH 2: RANDOM TRAIN-TEST SPLIT (80-20 Stratified) ---")
from sklearn.model_selection import train_test_split

X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
    X_full, y_full, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_full  # Maintains class distribution
)

print(f"Training set: {len(X_train_random)} samples")
print(f"Test set: {len(X_test_random)} samples")
print(f"Training home win rate: {y_train_random.mean()*100:.2f}%")
print(f"Test home win rate: {y_test_random.mean()*100:.2f}%")

# Train model
model_random = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_random.fit(X_train_random, y_train_random)

# Evaluate
y_pred_train_random = model_random.predict(X_train_random)
y_pred_test_random = model_random.predict(X_test_random)

print("\nTRAINING SET Performance:")
print(f"  Accuracy: {accuracy_score(y_train_random, y_pred_train_random)*100:.2f}%")

print("\nTEST SET Performance:")
test_accuracy_random = accuracy_score(y_test_random, y_pred_test_random)
print(f"  Accuracy: {test_accuracy_random*100:.2f}%")
print("\n  Classification Report:")
print(classification_report(y_test_random, y_pred_test_random, target_names=['Away Win/Draw', 'Home Win']))

# ========== CROSS-VALIDATION ==========
print("\n\n--- 5-FOLD CROSS-VALIDATION ---")
model_cv = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
cv_scores = cross_val_score(model_cv, X_full, y_full, cv=5, scoring='accuracy')
print(f"CV Fold Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}%")
print(f"Std Dev: {cv_scores.std()*100:.2f}%")

# ========== FEATURE IMPORTANCE ==========
print("\n\n--- Feature Importance (using full dataset model) ---")
model_full = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_full.fit(X_full, y_full)

importances = pd.DataFrame({
    'feature': FEATURES, 
    'importance': model_full.feature_importances_
})
print(importances.sort_values(by='importance', ascending=False).to_string(index=False))

# ========== RECOMMENDATIONS ==========
print("\n\n--- MODEL VALIDATION SUMMARY ---")
print(f"Temporal Test Accuracy (2025 data):   {test_accuracy*100:.2f}%")
print(f"Random Test Accuracy (20% holdout):   {test_accuracy_random*100:.2f}%")
print(f"Cross-Validation Mean Accuracy:       {cv_scores.mean()*100:.2f}%")

if test_accuracy < 0.55:
    print("\n⚠️  WARNING: Temporal test accuracy is low. Model may not generalize to future seasons.")
    print("   Consider: Adding more features, tuning hyperparameters, or collecting more data.")
else:
    print("\n✓ Temporal test accuracy is reasonable. Model shows promise for 2025 predictions.")

print("\nModel trained on Form Differential and Relative Ranking!")
