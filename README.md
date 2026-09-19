# Prem Predictor

A Premier League match outcome predictor built with Random Forest / XGBoost, trained on live match data pulled from a football API.

## Goal

Predict the outcome (Home Win / Draw / Away Win) of Premier League matches, then extend toward predicting exact scorelines. Longer term, publish predictions and rolling accuracy on a small web dashboard that updates after each matchweek.

## Roadmap

- **Phase 1 — Outcome classifier**: pull historical match data from a live API, engineer features (form, head-to-head, home/away splits, etc.), train and evaluate a Random Forest / XGBoost classifier on match outcome (H/D/A).
- **Phase 2 — Scoreline prediction**: extend or replace the classifier to predict exact scorelines (e.g. via separate home/away goal regressors).
- **Phase 3 — Web dashboard**: a page that refreshes after every matchweek, showing that week's results vs. predictions (with accuracy), and predictions for the upcoming matchweek.

## Project structure

```
data/raw/               untouched API pulls
data/raw/matches/       per-match h2h data, one file per match_id
data/processed/         cleaned / feature-engineered data
src/data_fetch.py       pulls standings, fixtures, and h2h from live-football-api.com
src/features.py         feature engineering
src/train.py            model training
src/predict.py          generate predictions
notebooks/              exploration / EDA
models/                 saved trained models
docs/DECISIONS.md       running log of design decisions and why they were made
```

## Setup

```
python -m venv prempredictor
source prempredictor/bin/activate
pip install -r requirements.txt
```

API credentials go in a local `.env` file (not committed, see `.env.example` for the expected variable name). Data source is [live-football-api.com](https://live-football-api.com) — see `docs/DECISIONS.md` for details on endpoints and gotchas (e.g. league IDs are opaque strings, not slugs).

## Status

- **Data fetching** (`src/data_fetch.py`) — pulls league standings, fixtures (looped week by week), and head-to-head data per match from live-football-api.com, saving raw JSON under `data/raw/`. Supports multiple seasons (`SEASONS` constant — current + 4 prior).
- **Feature engineering** (`src/features.py`) — builds a labeled training DataFrame (`build_training_set`) from all finished matches across the 5 seasons, and a features-only DataFrame for the next unplayed matchweek (`build_prediction_set`). Features are computed point-in-time (form, points, goal difference, head-to-head) to avoid leaking future information into historical rows.
- **Model training** (`src/train.py`) — scaffolded, not yet implemented.

See `docs/DECISIONS.md` for the full history, reasoning, and open questions.
