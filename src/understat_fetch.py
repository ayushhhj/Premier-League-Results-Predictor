import json
import os
from pathlib import PosixPath
import soccerdata as sd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "understat")
SEASONS = ["2026/2027", "2025/2026", "2024/2025", "2023/2024", "2022/2023"]


def _understat_season(season):
    """"2022/2023" -> "2022", soccerdata's Understat season format.

    A bare year can be ambiguous: "2021" resolved to 2020/21. Check the dates of what
    comes back when fetching an older season.
    """
    return season.split('/')[0]


def get_understat_matches(season):
    """Fetch a season's Understat team match stats and save them to data/raw/understat/<season>.json."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df = sd.Understat(leagues='ENG-Premier League', seasons=_understat_season(season)).read_team_match_stats().reset_index().to_dict(orient="records")
    path = os.path.join(DATA_DIR, f"{season.replace('/', '-')}.json")
    with open(path, 'w') as f:
      json.dump(df, f, default=str)

    return df



def get_all_understat_seasons(seasons):
    """get_understat_matches for each of the given seasons."""
    season_matches = []
    for season in seasons:
      season_matches.append(get_understat_matches(season))
    
    return season_matches


if __name__ == "__main__":
    get_all_understat_seasons(["2026/2027"])
