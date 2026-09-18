import requests
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FOOTBALL_API_KEY")
URL = "https://www.live-football-api.com/api/v1"


PREMIER_LEAGUE_ID = "2kwbbcootiqqgmrzs6o5inle5"


def get_league_standings():
    params = {
        "api_key": API_KEY,
        "league_id": PREMIER_LEAGUE_ID,
        "season": "2026/2027",
        "lang": "en",
    }
    response = requests.get(URL + "/league_standings", params=params)
    json.dump(response.json(), open(f"data/raw/league_standings.json", "w"))
    return response.json()

def get_league_matches():
    all_weeks = []
    for i in range(1, 39):
        params = {
            "api_key": API_KEY,
            "league_id": PREMIER_LEAGUE_ID,
            "season": "2026/2027",
            "week": i,
            "lang": "en",
        }
        response = requests.get(URL + "/league_fixtures", params=params)
        all_weeks.append(response.json())
        json.dump(all_weeks, open(f"data/raw/league_fixtures.json", "w"))
        time.sleep(0.5)
    return all_weeks

def extract_match_ids():
    match_ids = []
    all_weeks = json.load(open(f"data/raw/league_fixtures.json"))
    for week in all_weeks:
        for match in week["data"]["weeks"][0]["matches"]:
            match_id = match["id"]
            match_ids.append(match_id)
    return match_ids

    
def get_h2h_matches():
    match_ids = extract_match_ids()
    for match_id in match_ids:
        params = {
            "api_key": API_KEY,
            "match_id": match_id,
            "lang": "en",
        }
        response = requests.get(URL + "/h2h", params=params)
        json.dump(response.json(), open(f"data/raw/matches/h2h_{match_id}.json", "w"))
    return response.json()


if __name__ == "__main__":
    standings = get_league_standings()
    matches = get_league_matches()
    h2h_matches = get_h2h_matches()
    print(standings)
    print(matches)
    print(h2h_matches)