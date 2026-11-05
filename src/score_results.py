"""Score saved predictions against actual results.

Reads every data/predictions/<season>_GW<week>.csv and writes to data/results/:
- <season>_scored.csv: each predicted match with its score, result and whether the
  pick was correct (blank until played)
- <season>_summary.csv: played, correct, accuracy and the always-home baseline per
  gameweek, plus a Season row

Both files are rebuilt from scratch on every run.

Usage (from src/):
    python score_results.py
"""
import glob
import os
import sys

import pandas as pd

from features import SEASONS, flatten_fixtures

CURRENT_SEASON = SEASONS[0]
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "data", "predictions")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")


def season_tag(season):
    """"2026/2027" -> "2026-2027"."""
    return season.replace("/", "-")


def load_predictions(season=CURRENT_SEASON):
    """Every saved prediction for the season in one DataFrame, or None if there are none yet."""
    if os.path.exists(PREDICTIONS_DIR):
        files = glob.glob(os.path.join(PREDICTIONS_DIR, f"{season_tag(season)}_GW*.csv"))
        if files:
            dfs = [pd.read_csv(f) for f in files]
            return pd.concat(dfs, ignore_index=True).sort_values(by=["week", "date", "kickoff"])

    return None

def load_actual_results(season=CURRENT_SEASON):
    """match_id, home_score, away_score and result (H/D/A) for every finished match."""
    matches = []
    home_scores = []
    away_scores = []
    results = []
    for m in flatten_fixtures(season):
        if m["status"]["status"] == "finished":
            matches.append(m["id"])
            home_scores.append(int(m["home"]["score"]))
            away_scores.append(int(m["away"]["score"]))
            if int(m["home"]["score"]) > int(m["away"]["score"]):
                results.append("H")
            elif int(m["home"]["score"]) < int(m["away"]["score"]):
                results.append("A")
            else:
                results.append("D")

    return pd.DataFrame({
        "match_id": matches,
        "home_score": home_scores,
        "away_score": away_scores,
        "result": results,
    })


def score_predictions(predictions, results):
    """Left-join predictions to results, adding `played` and `correct` (False until played)."""
    score = predictions.merge(results, on="match_id", how="left")
    score["played"] = score["result"].notna()
    score["correct"] = (score["prediction"] == score["result"]) & (score["played"])
    return score


def summarize(scored):
    """Played, correct, accuracy and always-home baseline per gameweek, plus a Season row."""
    summary = scored[scored["played"]].groupby("week").agg(
        played=("match_id", "count"),
        correct=("correct", "sum"),
    )
    summary["accuracy"] = summary["correct"] / summary["played"]
    summary["home_baseline"] = (scored.loc[scored["played"], "result"] == "H").groupby(scored.loc[scored["played"], "week"]).mean()
    summary.loc["Season"] = summary.sum(numeric_only=True)
    summary.loc["Season", "accuracy"] = summary.loc["Season", "correct"] / summary.loc["Season", "played"]
    summary.loc["Season", "home_baseline"] = (scored.loc[scored["played"], "result"] == "H").mean()
    return summary


def save(scored, summary, season=CURRENT_SEASON):
    """Write both files to RESULTS_DIR (overwriting) and return their paths."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    season_tag_str = season_tag(season)
    scored_path = os.path.join(RESULTS_DIR, f"{season_tag_str}_scored.csv")
    summary_path = os.path.join(RESULTS_DIR, f"{season_tag_str}_summary.csv")
    scored.to_csv(scored_path, index=False)
    summary.reset_index().to_csv(summary_path, index=False)  
    return scored_path, summary_path


def main():
    """Command-line entry point. Returns the exit code."""
    predictions = load_predictions()
    if predictions is None:
        print("No predictions yet.")
        return 0

    results = load_actual_results()
    scored = score_predictions(predictions, results)
    summary = summarize(scored)
    save(scored, summary)

    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())