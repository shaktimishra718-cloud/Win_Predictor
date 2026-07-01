from google.colab import userdata
import requests
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier

def prediction(api_key, league_id, season , upcoming_only):
  url = "https://v3.football.api-sports.io/fixtures"
  headers = {"x-apisports-key":api_key}
  if upcoming_only:
    querystring = {"league" : league_id, "season" : season, "next" : "10"}
  else:
    querystring = {"league" : league_id , "season": season , "last": "50"}#Fetch last 50 match for training

  response = requests.get(url, headers = headers , params = querystring)
  if response.status_code != 200 or response.json().get("errors"):
        print(f"Error fetching data. Check your API key or limits.")
        return None
  for item in response.json().get("response" , []):
    matches.append({
        "fixture_id": item["fixture"]["id"],
        "home_team": item["teams"]["home"]["name"],
        "away_team": item["teams"]["away"]["name"],
        "home_goals": item["goals"]["home"],
        "away_goals": item["goals"]["away"]
      })
    return pd.DataFrame(matches)
userdata.get('APISPORTS_KEY')
