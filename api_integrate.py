def prediction(api_key, league_id, comp_name, is_neutral, season, upcoming_only):
    # Define the API endpoint and headers
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": api_key}
    
    if upcoming_only:
        querystring = {"league": league_id, "season": season, "next": "10"}
    else:
        querystring = {"league": league_id, "season": season, "last": "40"} # Pull last 40 games per league
        
    response = requests.get(url, headers=headers, params=querystring)
    
    if response.status_code != 200 or response.json().get("errors"):
        print(f"Skipping {comp_name}: API limit reached or invalid response.")
        return pd.DataFrame()
        
    matches = []
    for item in response.json().get("response", []):
        matches.append({
            "fixture_id": item["fixture"]["id"],
            "competition": comp_name,
            "home_team": item["teams"]["home"]["name"],
            "away_team": item["teams"]["away"]["name"],
            "is_neutral_venue": int(is_neutral), # 1 for True, 0 for False
            "home_goals": item["goals"]["home"],
            "away_goals": item["goals"]["away"]
        })
    return pd.DataFrame(matches)
