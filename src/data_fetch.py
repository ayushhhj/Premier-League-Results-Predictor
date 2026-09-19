import requests
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FOOTBALL_API_KEY")
URL = "https://www.live-football-api.com/api/v1"

PREMIER_LEAGUE_ID = "2kwbbcootiqqgmrzs6o5inle5"

# Current season plus 4 prior seasons, per the 5-season training window decision.
SEASONS = ["2026/2027", "2025/2026", "2024/2025", "2023/2024", "2022/2023"]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
MATCHES_DIR = os.path.join(DATA_DIR, "matches")


def _season_tag(season):
    return season.replace("/", "-")


def get_league_standings(season):
    params = {
        "api_key": API_KEY,
        "league_id": PREMIER_LEAGUE_ID,
        "season": season,
        "lang": "en",
    }
    response = requests.get(URL + "/league_standings", params=params)
    data = response.json()
    path = os.path.join(DATA_DIR, f"league_standings_{_season_tag(season)}.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def get_league_matches(season):
    all_weeks = []
    path = os.path.join(DATA_DIR, f"league_fixtures_{_season_tag(season)}.json")
    for i in range(1, 39):
        params = {
            "api_key": API_KEY,
            "league_id": PREMIER_LEAGUE_ID,
            "season": season,
            "week": i,
            "lang": "en",
        }
        response = requests.get(URL + "/league_fixtures", params=params)
        all_weeks.append(response.json())
        with open(path, "w") as f:
            json.dump(all_weeks, f)
        time.sleep(0.5)
    return all_weeks


def extract_match_ids(season):
    path = os.path.join(DATA_DIR, f"league_fixtures_{_season_tag(season)}.json")
    with open(path) as f:
        all_weeks = json.load(f)

    match_ids = []
    for week in all_weeks:
        if "data" not in week:
            continue
        for match in week["data"]["weeks"][0]["matches"]:
            match_ids.append(match["id"])
    return match_ids


def get_h2h_matches(season):
    os.makedirs(MATCHES_DIR, exist_ok=True)
    match_ids = extract_match_ids(season)
    results = []
    for match_id in match_ids:
        params = {
            "api_key": API_KEY,
            "match_id": match_id,
            "lang": "en",
        }
        response = requests.get(URL + "/h2h", params=params)
        data = response.json()
        results.append(data)
        path = os.path.join(MATCHES_DIR, f"h2h_{match_id}.json")
        with open(path, "w") as f:
            json.dump(data, f)
        time.sleep(0.5)
    return results


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    for season in SEASONS:
        standings = get_league_standings(season)
        matches = get_league_matches(season)
        print(f"{season}: standings + fixtures saved")
    # h2h_matches = get_h2h_matches(SEASONS[0])
    # print(f"H2H matches saved for {SEASONS[0]}")