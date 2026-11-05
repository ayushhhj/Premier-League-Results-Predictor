"""Predict the next gameweek with ordered logistic regression.

Trains on every finished match on disk and writes
data/predictions/<season>_GW<week>.csv. Doesn't fetch data, so refresh it first
(data_fetch.py, get_all_team_matches). Saved predictions are locked: an existing file
is never overwritten.

Usage (from src/):
    python predict.py                       # next gameweek
    python predict.py --week 11             # a specific gameweek
    python predict.py --if-starts-tomorrow  # only if the next gameweek starts tomorrow
"""
import argparse
import os
import sys
from datetime import date, datetime, timedelta

import pandas as pd

from features import SEASONS, build_prediction_set, build_training_set, flatten_fixtures
from train import FEATURE_COLUMNS, train_ordered_logit

CURRENT_SEASON = SEASONS[0]
PREDICTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "predictions")
MODEL_NAME = "Ordered Logit"
# Columns written to the CSV; features are left out
OUTPUT_COLUMNS = [
    "season", "week", "match_id", "date", "kickoff", "home_team", "away_team",
    "p_home", "p_draw", "p_away", "prediction", "model", "generated_at",
]


def next_gameweek(season_matches, today):
    """(week, date of its first match) for the next gameweek, or (None, None) after the season.

    Only unplayed matches dated today or later count, so a postponed match can't hold
    the "next" gameweek back.
    """
    unplayed = [m for m in season_matches if m["status"]["status"] != "finished" and m["date"] >= today.isoformat()]
    if unplayed:
        week = min(m["week"] for m in unplayed)
        first_date = date.fromisoformat(min(m["date"] for m in unplayed if m["week"] == week))
        return week, first_date
    else:
        return None, None  


def should_run_today(first_date, today):
    """True if the gameweek's first match is tomorrow."""
    return first_date == today + timedelta(days=1)


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
    season_tag = CURRENT_SEASON.replace("/", "-")
    filename = f"{season_tag}_GW{week}.csv"
    path = os.path.join(PREDICTIONS_DIR, filename)
    if os.path.exists(path):
        # locked: delete the file by hand to redo a gameweek
        print(f"{path} already exists — not overwriting.")
        return None
    predictions.to_csv(path, index=False)
    return path


def main():
    """Command-line entry point. Returns the exit code."""
    today = date.today()
    week, first_date = next_gameweek(flatten_fixtures(CURRENT_SEASON), today)
    parser = argparse.ArgumentParser(description="Predict the next Premier League gameweek.")
    parser.add_argument("--week", type=int, help="Gameweek to predict (default: next unplayed)")
    parser.add_argument("--if-starts-tomorrow", action="store_true", help="Only run if the gameweek's first match is tomorrow")
    args = parser.parse_args()

    if args.week:
        week = args.week
        first_date = None  

    if not week:
            print("Season over.")
            return 0
    
    if args.if_starts_tomorrow and not should_run_today(first_date, today):
        print("Skipping run: the gameweek's first match is not tomorrow.")
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
