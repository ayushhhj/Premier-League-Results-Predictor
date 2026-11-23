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


RETRIES = 3
RETRY_WAIT = 5  # seconds, doubled after each failed attempt


def _season_tag(season):
    return season.replace("/", "-")


def warn(message):
    """Print a warning; GitHub Actions also shows it in the run summary."""
    print(f"::warning::{message}")


def get_json(endpoint, params):
    """GET an API endpoint and return the parsed response, or None if it still fails after retries.

    A response only counts if it's JSON with a "data" field and success isn't false; rate
    limits and outages come back as errors or empty bodies.
    """
    wait = RETRY_WAIT
    for attempt in range(1, RETRIES + 1):
        problem = None
        try:
            response = requests.get(URL + endpoint, params=params, timeout=30)
            payload = response.json()
            if response.ok and isinstance(payload, dict) and "data" in payload and payload.get("success") is not False:
                return payload
            problem = f"HTTP {response.status_code}: {str(payload.get('message') if isinstance(payload, dict) else payload)[:100]}"
        except (requests.RequestException, ValueError) as e:  # ValueError covers non-JSON bodies
            problem = type(e).__name__
        if attempt < RETRIES:
            time.sleep(wait)
            wait *= 2
    # params holds the API key, so only the endpoint goes in the message
    warn(f"{endpoint} failed after {RETRIES} attempts ({problem})")
    return None


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def get_league_standings(season):
    params = {
        "api_key": API_KEY,
        "league_id": PREMIER_LEAGUE_ID,
        "season": season,
        "lang": "en",
    }
    data = get_json("/league_standings", params)
    if data is None:
        warn(f"{season} standings: kept the previously saved file")
        return None
    save_json(os.path.join(DATA_DIR, f"league_standings_{_season_tag(season)}.json"), data)
    return data


def get_league_matches(season):
    """Fetch all 38 gameweeks of fixtures. A gameweek that fails keeps its previously saved data."""
    path = os.path.join(DATA_DIR, f"league_fixtures_{_season_tag(season)}.json")
    try:
        with open(path) as f:
            previous = json.load(f)
    except (FileNotFoundError, ValueError):
        previous = []

    all_weeks = []
    for i in range(1, 39):
        params = {
            "api_key": API_KEY,
            "league_id": PREMIER_LEAGUE_ID,
            "season": season,
            "week": i,
            "lang": "en",
        }
        week = get_json("/league_fixtures", params)
        if week is None:
            week = previous[i - 1] if len(previous) >= i else {}
            warn(f"{season} gameweek {i}: kept the previously saved fixtures")
        all_weeks.append(week)
        time.sleep(0.5)
    save_json(path, all_weeks)
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
        data = get_json("/h2h", params)
        if data is not None:
            results.append(data)
            save_json(os.path.join(MATCHES_DIR, f"h2h_{match_id}.json"), data)
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
    """Fetch and save a team's matches in all competitions for a season (used for rest days).

    Returns None and keeps the previously saved file if the request fails.
    """
    params = {
        "api_key": API_KEY,
        "team_id": team_id,
        "season": season,
        "lang": "en",
    }
    data = get_json("/team_matches", params)
    if data is None:
        return None
    os.makedirs(TEAM_MATCHES_DIR, exist_ok=True)
    save_json(os.path.join(TEAM_MATCHES_DIR, f"{team_id}_{_season_tag(season)}.json"), data)
    return data


def get_all_team_matches(seasons=SEASONS):
    """Fetch get_team_matches for every team in each season's fixtures (~20 API calls per season)."""
    for season in seasons:
        team_ids = get_season_team_ids(season)
        failed = 0
        for team_id in team_ids:
            if get_team_matches(team_id, season) is None:
                failed += 1
            time.sleep(0.5)
        print(f"{season}: team_matches saved for {len(team_ids) - failed} of {len(team_ids)} teams")
        if failed:
            warn(f"{season}: {failed} team(s) kept their previously saved matches")


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
