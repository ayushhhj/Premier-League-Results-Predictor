"""Predict the next gameweek with ordered logistic regression.

Trains on every finished match on disk and writes
data/predictions/<season>_GW<week>.csv. Doesn't fetch data, so refresh it first
(data_fetch.py, get_all_team_matches). Saved predictions are locked: an existing file
is never overwritten.

Usage (from src/):
    python predict.py                       # next gameweek
    python predict.py --week 11             # a specific gameweek
    python predict.py --if-due              # only if the next gameweek is due (see is_due)
"""
import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from features import SEASONS, build_prediction_set, build_training_set, flatten_fixtures
from train import FEATURE_COLUMNS, train_ordered_logit

CURRENT_SEASON = SEASONS[0]
PREDICTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "predictions")
MODEL_NAME = "Ordered Logit"
KICKOFF_TZ = ZoneInfo("Europe/London")  # fixture dates and kickoff times are UK local time
PREDICT_WINDOW = timedelta(hours=48)    # how long before the first kickoff predictions can be made
# Columns written to the CSV; features are left out
OUTPUT_COLUMNS = [
    "season", "week", "match_id", "date", "kickoff", "home_team", "away_team",
    "p_home", "p_draw", "p_away", "prediction", "model", "generated_at",
]


def kickoff_time(match):
    """A match's kickoff as a UTC datetime. Unknown times count as midnight, i.e. early."""
    try:
        hour, minute = (int(x) for x in match["kickoff"].split(":"))
    except (AttributeError, ValueError):
        hour, minute = 0, 0
    local = datetime.combine(date.fromisoformat(match["date"]), datetime.min.time()).replace(
        hour=hour, minute=minute, tzinfo=KICKOFF_TZ)
    return local.astimezone(timezone.utc)


def next_gameweek(season_matches, today):
    """(week, first kickoff as UTC datetime) for the next gameweek, or (None, None) after the season.

    Only unplayed matches dated today or later count, so a postponed match can't hold
    the "next" gameweek back.
    """
    unplayed = [m for m in season_matches if m["status"]["status"] != "finished" and m["date"] >= today.isoformat()]
    if unplayed:
        week = min(m["week"] for m in unplayed)
        first_kickoff = min(kickoff_time(m) for m in unplayed if m["week"] == week)
        return week, first_kickoff
    else:
        return None, None


def predictions_path(week):
    return os.path.join(PREDICTIONS_DIR, f"{CURRENT_SEASON.replace('/', '-')}_GW{week}.csv")


def is_due(week, first_kickoff, now):
    """True if the gameweek has no saved predictions, its first unplayed kickoff is still
    ahead, and that kickoff is within PREDICT_WINDOW.

    Any run inside the window can make the prediction, so a late or missed run is
    caught up by the next one. If every run before the gameweek starts is missed, a run
    between matchdays still predicts the matches that haven't kicked off.
    """
    if os.path.exists(predictions_path(week)):
        return False
    return now < first_kickoff <= now + PREDICT_WINDOW


def train_model():
    """Fit ordered logit on every finished match. Returns (model, number of training rows)."""
    train_df = build_training_set(SEASONS)
    fitted_model = train_ordered_logit(train_df)
    return fitted_model, len(train_df)


def predict_week(model, week):
    """Predictions for a gameweek's unplayed matches, sorted by kickoff, as OUTPUT_COLUMNS.

    Ordered logit returns probabilities in [A, D, H] order.
    """
    pred_df = build_prediction_set(CURRENT_SEASON, week=week)
    if pred_df.empty:
        return pred_df
    probabilities = model.model.predict(model.params, exog=pred_df[FEATURE_COLUMNS])

    lookup = {m["id"]: m for m in flatten_fixtures(CURRENT_SEASON)}
    pred_df["week"] = week
    pred_df["p_home"] = probabilities[:, 2]  
    pred_df["p_draw"] = probabilities[:, 1]  
    pred_df["p_away"] = probabilities[:, 0] 
    pred_df["prediction"] = pd.Series(probabilities.argmax(axis=1), index=pred_df.index).map({0: "A", 1: "D", 2: "H"})
    pred_df["model"] = MODEL_NAME
    pred_df["generated_at"] = datetime.now().isoformat(timespec="seconds")
    pred_df["kickoff"] = pred_df["match_id"].map(lambda mid: lookup[mid]["kickoff"])
    pred_df["home_team"] = pred_df["match_id"].map(lambda mid: lookup[mid]["home"]["name"])
    pred_df["away_team"] = pred_df["match_id"].map(lambda mid: lookup[mid]["away"]["name"])
    pred_df = pred_df.sort_values(by=["date", "kickoff"])
    return pred_df[OUTPUT_COLUMNS]


def save_predictions(predictions, week):
    """Save predictions and return the path, or None without writing if the gameweek already has a file."""
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    path = predictions_path(week)
    if os.path.exists(path):
        # locked: delete the file by hand to redo a gameweek
        print(f"{path} already exists — not overwriting.")
        return None
    predictions.to_csv(path, index=False)
    return path


def main():
    """Command-line entry point. Returns the exit code."""
    now = datetime.now(timezone.utc)
    week, first_kickoff = next_gameweek(flatten_fixtures(CURRENT_SEASON), now.date())
    parser = argparse.ArgumentParser(description="Predict the next Premier League gameweek.")
    parser.add_argument("--week", type=int, help="Gameweek to predict (default: next unplayed)")
    parser.add_argument("--if-due", action="store_true", help="Only run if the next gameweek is due (see is_due)")
    args = parser.parse_args()

    if args.week:
        week = args.week

    if not week:
            print("Season over.")
            return 0

    if args.if_due and not args.week and not is_due(week, first_kickoff, now):
        print(f"GW{week} not due: first kickoff {first_kickoff:%a %d %b %H:%M} UTC; "
              f"predictions {'already saved' if os.path.exists(predictions_path(week)) else 'not saved yet'}.")
        return 0

    model, n_train = train_model()
    predictions = predict_week(model, week)
    if predictions.empty:
        print("No predictions to save.")
        return 0
    else:
        print(f"Predictions for {CURRENT_SEASON} GW{week} (trained on {n_train} matches):")
        print(predictions[["date", "home_team", "away_team", "p_home", "p_draw", "p_away", "prediction"]])

    path = save_predictions(predictions, week)
    if path is not None:
        print(f"Predictions saved to {path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
