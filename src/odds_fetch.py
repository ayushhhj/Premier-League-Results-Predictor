import json
import os
import soccerdata as sd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "odds")


def _odds_season(season):
    """"2022/2023" -> "2022", soccerdata's season format."""
    return season.split("/")[0]


def get_odds(season):
    """Fetch a season's match odds from Football-Data.co.uk and save them to data/raw/odds/<season>.json.

    Keeps the date, teams and AvgH/AvgD/AvgA (average 1X2 odds across bookmakers).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    df = sd.MatchHistory(leagues="ENG-Premier League", seasons=_odds_season(season)).read_games()
    df = df.reset_index()[["date", "home_team", "away_team", "AvgH", "AvgD", "AvgA"]]
    records = df.to_dict(orient="records")

    path = os.path.join(DATA_DIR, f"{season.replace('/', '-')}.json")
    with open(path, "w") as f:
        json.dump(records, f, default=str)

    return records


def get_all_odds_seasons(seasons):
    """Call get_odds for every season in seasons."""
    return [get_odds(season) for season in seasons]


if __name__ == "__main__":
    pass
