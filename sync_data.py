import pandas as pd
import time
from sqlalchemy import text
from database import get_engine
from api_integrate import fetch_matches, TARGET_COMPETITIONS, SLEEP_TIME

def sync_historical_data():
    engine = get_engine()
    if not engine:
        print("Database connection failed. Exiting.")
        return

    print("\n" + "="*45)
    print(" 🌍 MULTI-TABLE DATABASE SYNC TERMINAL 🌍 ")
    print("="*45)
    
    # 1. Create the Interactive Menu
    comp_names = list(TARGET_COMPETITIONS.keys())
    for i, name in enumerate(comp_names, 1):
        print(f" [{i}] {name}")
    print(f" [{len(comp_names) + 1}] Sync ALL (Builds/Updates All 3 Tables)")
    
    choice = input("\nEnter the number to sync (e.g., 1): ").strip()
    
    # 2. Figure out what the user chose
    try:
        choice_idx = int(choice) - 1
        if choice_idx == len(comp_names):
            selected_comps = TARGET_COMPETITIONS # User chose ALL
            print("\nInitiating full multi-table sync...")
        else:
            comp_name = comp_names[choice_idx]
            selected_comps = {comp_name: TARGET_COMPETITIONS[comp_name]} # User chose ONE
            print(f"\nInitiating sync for {comp_name} only...")
    except (ValueError, IndexError):
        print("Invalid selection. Exiting.")
        return

    historical_dfs = []

    # 3. Fetch ONLY the selected data
    for comp_name, config in selected_comps.items():
        for season in config["seasons"]:
            print(f" Fetching {comp_name} - Season {season}...")
            df = fetch_matches(config["code"], season=season, upcoming_only=False)
            
            if not df.empty:
                df['competition'] = comp_name
                df['season'] = season
                df['is_neutral_venue'] = int(config["neutral_venues"])
                historical_dfs.append(df)
            
            time.sleep(SLEEP_TIME) # Respect the rate limit!

    if not historical_dfs:
        print("No data fetched. Check your API key, connection, or rate limits.")
        return

    master_history = pd.concat(historical_dfs, ignore_index=True)
    print(f"\nTotal matches fetched from API: {len(master_history)}")

    # 4. Standardize Columns & Create IDs
    rename_map = {'id': 'match_id', 'fixture_id': 'match_id', 'home_goals': 'home_score', 'away_goals': 'away_score'}
    master_history = master_history.rename(columns=rename_map)

    if 'match_id' not in master_history.columns:
        import hashlib
        def create_match_id(row):
            unique_string = f"{row.get('match_date')}_{row.get('home_team')}_{row.get('away_team')}"
            return int(hashlib.md5(unique_string.encode()).hexdigest(), 16) % (10**10)
        master_history['match_id'] = master_history.apply(create_match_id, axis=1)

    db_columns = ['match_id', 'competition', 'season', 'match_date', 'home_team', 'away_team', 'home_score', 'away_score', 'is_neutral_venue']
    final_cols = [col for col in db_columns if col in master_history.columns]
    master_history = master_history[final_cols]

    # THE 2026 FIX: Drop any matches that haven't happened yet (score is blank/NaN)
    master_history = master_history.dropna(subset=['home_score', 'away_score'])

    # 5. Dynamic Table Routing (Strict Isolation)
    print("\nWriting to Isolated PostgreSQL Tables...")
    try:
        # Loop through the competitions we just fetched
        for comp_name, config in selected_comps.items():
            
            # Create a clean table name (e.g., matches_pl, matches_pd, matches_cl)
            table_name = f"matches_{config['code'].lower()}"
            
            # Filter the massive dataframe down to ONLY this specific competition
            comp_data = master_history[master_history['competition'] == comp_name]
            
            if not comp_data.empty:
                print(f" -> Routing {len(comp_data)} matches to table: '{table_name}'...")
                
                # Because tables are completely isolated, we can safely 'replace' them 
                # without accidentally deleting a different league's data!
                comp_data.to_sql(table_name, engine, if_exists='replace', index=False)
                
        print("\nSuccess! Data routed and synchronized perfectly.")
    except Exception as e:
        print(f"Database insertion failed: {e}")

if __name__ == "__main__":
    sync_historical_data()