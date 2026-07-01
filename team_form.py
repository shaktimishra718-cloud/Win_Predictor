#FEATURE ENGINEERING FUNCTION
def feature_engineering(df , standing_dict ):
  # 1. Sort chronologically so our rolling math calculates past-to-present
    df = df.sort_values(by='match_date').reset_index(drop=True)
 
  # 2. Define the Target Variable (1 if Home Win, 0 otherwise)
    df['home_win'] = (df['home_goals'] > df['away_goals']).astype(float)
    df['away_win'] = (df['away_goals'] > df['home_goals']).astype(float)
  
  # 3. Calculate Rolling Form (Win % over the last 5 Home/Away games)
    df['home_form'] = df.groupby('home_team')['home_win'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df['away_form'] = df.groupby('away_team')['away_win'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        
  # Early season form
    df['home_form'] = df['home_form'].fillna(0.5)
    df['away_form'] = df['away_form'].fillna(0.5)

  # Mapping the League Table
    df['home_rank'] = df['home_team'].map(standings_dict).fillna(10) # Default to mid-table if unknown
    df['away_rank'] = df['away_team'].map(standings_dict).fillna(10)
    
    return df
