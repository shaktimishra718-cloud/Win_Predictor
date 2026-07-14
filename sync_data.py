import pandas as pd
import time
from database import get_engine
from api_integrate import fetch_matches, TARGET_COMPETITIONS, SLEEP_TIME

def sync_historical_data():
    print("Initiating full historical data migration from API to PostgreSQL...")
    engine = get_engine()
    
    if not engine:
        print("Database connection failed. Exiting.")
        return

    historical_dfs = []

    # Loop through the competitions and seasons you defined in api_integrate.py
    for comp_name, config in TARGET_COMPETITIONS.items():
        for season in config["seasons"]:
            print(f"Fetching {comp_name} - Season {season}...")
            
            # Fetch past matches (upcoming_only=False)
            df = fetch_matches(config["code"], season=season, upcoming_only=False)
            
            if not df.empty:
                df['competition'] = comp_name
                df['season'] = season
                df['is_neutral_venue'] = int(config["neutral_venues"])
                historical_dfs.append(df)
            
            # Respect API rate limits
            time.sleep(SLEEP_TIME)

    if not historical_dfs:
        print("No data fetched. Check your API key or connection.")
        return

    # Combine all fetched seasons into one massive DataFrame
    master_history = pd.concat(historical_dfs, ignore_index=True)
    print(f"\nTotal historical matches fetched from API: {len(master_history)}")

    # Combine all fetched seasons into one massive DataFrame
    master_history = pd.concat(historical_dfs, ignore_index=True)
    print(f"\nTotal historical matches fetched from API: {len(master_history)}")

    # 1. THE TRANSLATION LAYER: Map your API column names to the Database column names
    rename_map = {
        'id': 'match_id',
        'fixture_id': 'match_id',
        'home_goals': 'home_score',
        'away_goals': 'away_score'
    }
    master_history = master_history.rename(columns=rename_map)

    # 2. THE ID GENERATOR: If your API didn't provide an ID, create a unique, stable one
    if 'match_id' not in master_history.columns:
        import hashlib
        def create_match_id(row):
            # Combine date and teams into a single string
            unique_string = f"{row.get('match_date')}_{row.get('home_team')}_{row.get('away_team')}"
            # Hash it into a unique 10-digit integer
            return int(hashlib.md5(unique_string.encode()).hexdigest(), 16) % (10**10)
        
        print("Synthesizing unique match IDs...")
        master_history['match_id'] = master_history.apply(create_match_id, axis=1)

    # 3. THE FILTER: Isolate only the columns that belong in the database
    db_columns = ['match_id', 'competition', 'season', 'match_date', 'home_team', 'away_team', 'home_score', 'away_score', 'is_neutral_venue']
    
    final_cols = [col for col in db_columns if col in master_history.columns]
    master_history = master_history[final_cols]

    print("Writing true historical data to PostgreSQL...")
    try:
        # push to database
        master_history.to_sql('matches', engine, if_exists='append', index=False)
        print("Success! Data perfectly synchronized to PostgreSQL.")
    except Exception as e:
        print(f"Database insertion failed: {e}")

if __name__ == "__main__":
    sync_historical_data()