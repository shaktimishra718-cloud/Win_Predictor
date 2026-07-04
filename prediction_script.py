# =====================================================================
# CELL 4: THE INFERENCE ENGINE (Predicting Future Matches)
# =====================================================================
import time
import pandas as pd

print("--- Phase 4: Fetching Upcoming Fixtures ---")
upcoming_matches = []

for comp_name, config in TARGET_COMPETITIONS.items():
    current_season = max(config["seasons"]) # Always predict for the latest season
    previous_season = current_season - 1    # Fallback season
    
    print(f"Fetching SCHEDULED matches for {comp_name} (Season {current_season})...")
    
    df = fetch_matches(config["code"], season=current_season, upcoming_only=True)
    
    if not df.empty:
        # 1. Fetch current standings
        standings_map = get_real_standings(config["code"], season=current_season)
        
        # 🚨 THE FIX: Cross-Season Warm-Up
        is_brand_new_season = len(standings_map) == 0 or all(v.get('form') == 0.5 or not v.get('form') for v in standings_map.values())
        
        if is_brand_new_season:
            print(f"   ⚠️ Season {current_season} has no history yet. Priming form features with final Season {previous_season} data...")
            standings_map = get_real_standings(config["code"], season=previous_season)
        
        df['competition'] = comp_name
        df['season'] = current_season
        df['is_neutral_venue'] = int(config["neutral_venues"])
        
        # Map current standings data
        df['home_rank'] = df['home_team'].apply(lambda x: standings_map.get(x, {}).get('rank', 10))
        df['away_rank'] = df['away_team'].apply(lambda x: standings_map.get(x, {}).get('rank', 10))
        df['home_form'] = df['home_team'].apply(lambda x: standings_map.get(x, {}).get('form', 0.5))
        df['away_form'] = df['away_team'].apply(lambda x: standings_map.get(x, {}).get('form', 0.5))
        df['form_diff'] = df['home_form'] - df['away_form']
        
        upcoming_matches.append(df)
        
    time.sleep(SLEEP_TIME)

if upcoming_matches:
    upcoming_df = pd.concat(upcoming_matches, ignore_index=True)
    
    print("\nConnecting future fixtures to historical timeline to calculate H2H and Fatigue...")
    
    # TRICK: Flag historical vs future data so we can separate them later
    master_history['is_future'] = False
    upcoming_df['is_future'] = True
    
    # Combine, sort chronologically, and calculate advanced features
    combined_timeline = pd.concat([master_history, upcoming_df], ignore_index=True)
    combined_timeline['match_date'] = pd.to_datetime(combined_timeline['match_date'])
    combined_timeline = combined_timeline.sort_values('match_date').reset_index(drop=True)
    
    # Run the feature engineering!
    combined_timeline = compute_h2h_and_fatigue(combined_timeline, h2h_n=5, default_rest_days=7)
    
    # Extract only the future matches back out
    inference_data = combined_timeline[combined_timeline['is_future'] == True].copy()
    
    # Drop rows where we couldn't calculate features
    inference_data = inference_data.dropna(subset=FEATURES)
    
    if not inference_data.empty:
        print(f"✅ Successfully engineered features for {len(inference_data)} upcoming matches!\n")
        
        X_infer = inference_data[FEATURES]
        
        # THE PREDICTION
        probabilities = model_temporal.predict_proba(X_infer)
        
        inference_data['prob_away_draw'] = probabilities[:, 0]
        inference_data['prob_home_win'] = probabilities[:, 1]
        
        # --- DASHBOARD OUTPUT ---
        print("=========================================================================")
        print("                 🔮 UPCOMING MATCH PREDICTIONS 🔮")
        print("=========================================================================")
        
        # Sort by date so we see the closest games first
        inference_data = inference_data.sort_values('match_date')
        
        for idx, row in inference_data.iterrows():
            home = row['home_team']
            away = row['away_team']
            date = row['match_date'].strftime('%A, %b %d at %H:%M')
            comp = row['competition']
            
            p_home = row['prob_home_win'] * 100
            p_away_draw = row['prob_away_draw'] * 100
            
            if p_home > 50:
                prediction = f"HOME WIN ({home})"
                confidence = p_home
                symbol = "🟢" if p_home > 65 else "🟡"
            else:
                prediction = f"AWAY/DRAW ({away} avoids defeat)"
                confidence = p_away_draw
                symbol = "🔴" if p_away_draw > 65 else "🟡"
            
            print(f"[{comp}] {date}")
            print(f"{home} vs {away}")
            print(f"{symbol} Prediction: {prediction} (Confidence: {confidence:.1f}%)")
            
            # Show the human why the model made this choice (Notice Form Diff is no longer 0.00!)
            print(f"   ↳ Form Diff: {row['form_diff']:.2f} | Rest Diff: {row['fatigue_diff']} days | H2H Home Win Rate: {row['h2h_home_win_rate']*100:.0f}%")
            print("-" * 73)
            
    else:
        print("Could not generate features for upcoming matches (data might be incomplete).")
else:
    print("No upcoming SCHEDULED matches found. (Is it the off-season or an international break?)")
