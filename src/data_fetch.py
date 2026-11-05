import requests
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FOOTBALL_API_KEY")
URL = "https://www.live-football-api.com/api/v1"

PREMIER_LEAGUE_ID = "2kwbbcootiqqgmrzs6o5inle5"

# Current season plus the 4 before it
SEASONS = ["2026/2027", "2025/2026", "2024/2025", "2023/2024", "2022/2023"]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
MATCHES_DIR = os.path.join(DATA_DIR, "matches")
TEAM_MATCHES_DIR = os.path.join(DATA_DIR, "team_matches")


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


def get_season_team_ids(season):
    """Team IDs appearing in a season's saved fixtures."""
    path = os.path.join(DATA_DIR, f"league_fixtures_{_season_tag(season)}.json")
    with open(path) as f:
        all_weeks = json.load(f)

    team_ids = set()
    for week in all_weeks:
        if "data" not in week:
            continue
        for match in week["data"]["weeks"][0]["matches"]:
            team_ids.add(match["home"]["id"])
            team_ids.add(match["away"]["id"])
    return team_ids


def get_team_matches(team_id, season):
    """Fetch and save a team's matches across all competitions for a season (used for rest days)."""
    params = {
        "api_key": API_KEY,
        "team_id": team_id,
        "season": season,
        "lang": "en",
    }
    response = requests.get(URL + "/team_matches", params=params)
    data = response.json()
    os.makedirs(TEAM_MATCHES_DIR, exist_ok=True)
    path = os.path.join(TEAM_MATCHES_DIR, f"{team_id}_{_season_tag(season)}.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def get_all_team_matches(seasons=SEASONS):
    """Fetch get_team_matches for every team in each season's fixtures (~20 API calls per season)."""
    for season in seasons:
        team_ids = get_season_team_ids(season)
        for team_id in team_ids:
            get_team_matches(team_id, season)
            time.sleep(0.5)
        print(f"{season}: team_matches saved for {len(team_ids)} teams")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    standings = get_league_standings("2026/2027")
    matches = get_league_matches("2026/2027")
    print(f"{"2026/2027"}: standings + fixtures saved")

    # One-off fetches, already run:
    # get_h2h_matches(SEASONS[0])           # h2h for the current season's fixtures
    # get_league_standings("2021/2022")     # 2021/22: Elo warm-up season
    # get_league_matches("2021/2022")
    # get_all_team_matches(["2021/2022"])
