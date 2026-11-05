# Prem Predictor

Predicts Premier League results (home win, draw or away win) one gameweek at a time. The model uses team stats calculated only from matches played before each fixture: form, home and away records, head-to-head results and rest days. It's trained on five seasons of matches.

## Goal

Predict every Premier League match one gameweek at a time, then move on to exact scorelines. Predictions and running accuracy go on a web page that updates itself each gameweek.

## Roadmap

- **Phase 1, outcome classifier (done):** data pipeline, point-in-time features, model comparison and a weekly prediction script.
- **Phase 2, scoreline prediction:** predict exact scores. Dixon-Coles (already in `train.py`) is the starting point because it models goal rates directly.
- **Phase 3, web dashboard (in progress):** a GitHub Actions job refreshes the data every day, predicts the next gameweek the day before it starts, scores past predictions and publishes a page on GitHub Pages.

## Model

The production model in `predict.py` is an ordered logistic regression. It treats away win, draw and home win as points on one scale, which tree models can't use.

It uses 11 features, each calculated only from matches before the one being predicted:

| Feature | What it is |
|---|---|
| `home_form`, `away_form` | Recency-weighted points from the last 5 league matches (decay 0.8) |
| `home_venue_form`, `away_venue_form` | The same, counting only home matches or only away matches |
| `home_venue_points`, `away_venue_points` | Points at that venue so far this season |
| `home_venue_goal_diff`, `away_venue_goal_diff` | Goal difference at that venue so far this season |
| `h2h_score` | Head-to-head record between the two clubs up to the match date |
| `home_rest_days`, `away_rest_days` | Days since each team's last competitive match in any competition |

Elo ratings, xG and bookmaker odds (with the margin removed) are also calculated for every match, but the model doesn't use them. Elo didn't help consistently. xG was dropped in November 2026 because it added almost nothing and its source, Understat, stopped returning data. Odds are left out on purpose so the accuracy reflects the model's own features; they serve as a benchmark when evaluating.

### How well it does

- Past seasons, walk-forward (trained on earlier seasons and tested on each of 2023/24, 2024/25 and 2025/26, 1,140 matches): 51.4%.
- 2026/27 so far, gameweek by gameweek (`holdout_gameweek.py --rolling`, GW1 to GW9, 90 matches, each gameweek predicted from earlier data only): 43.3% (39 of 90). The bookmakers' favourites got 47.8% and always picking the home team got 40%.
- Known weaknesses: the linear models never predict a draw, and gameweek 1 has almost nothing to go on because the season-based features reset to zero.

`docs/DECISIONS.md` records every experiment, including the ones that were rejected.

## Project structure

```
data/raw/                 raw API pulls and scraped data (JSON)
data/raw/matches/         per-match head-to-head data, one file per match_id
data/raw/team_matches/    each team's matches in all competitions (for rest days)
data/raw/understat/       xG per match, per season
data/raw/odds/            average pre-match 1X2 odds, per season
data/predictions/         one CSV of predictions per gameweek
data/results/             predictions scored against results, plus a per-gameweek summary
src/data_fetch.py         standings, fixtures, h2h and team matches from live-football-api.com
src/understat_fetch.py    xG from Understat (via soccerdata)
src/odds_fetch.py         historical odds from Football-Data.co.uk (via soccerdata)
src/features.py           point-in-time feature engineering, Elo, odds lookup
src/train.py              model training, evaluation, walk-forward tuning (experiment harness)
src/predict.py            predicts the next gameweek with Ordered Logit, writes data/predictions/
src/score_results.py      scores saved predictions against results, writes data/results/
src/build_site.py         builds the dashboard page (site/index.html)
src/holdout_gameweek.py   current-season test: hold out a gameweek, train on everything before it
.github/workflows/        daily GitHub Actions job: refresh, predict, score, build, publish
notebooks/                exploration / EDA
docs/DECISIONS.md         running log of design decisions and why they were made
```

## Setup

Needs Python 3.12 or newer, because `data_fetch.py` nests quotes inside an f-string.

```
python -m venv prempredictor
source prempredictor/bin/activate
pip install -r requirements.txt
```

The API key goes in a local `.env` file, which isn't committed (`.env.example` shows the variable name). The main data source is [live-football-api.com](https://live-football-api.com). `docs/DECISIONS.md` covers its endpoints and quirks, such as league IDs being opaque strings rather than readable names.

For the scheduled job, add the same key as a `FOOTBALL_API_KEY` repository secret on GitHub and set the Pages source to "GitHub Actions".

## Usage

Run everything from `src/`.

Refresh the current season's data before predicting, so the latest results are included:

```
python data_fetch.py                     # standings + all 38 weeks of fixtures
python -c "from data_fetch import get_all_team_matches; get_all_team_matches(['2026/2027'])"
python -c "from odds_fetch import get_odds; get_odds('2026/2027')"   # optional, evaluation only
```

Predict the next gameweek:

```
python predict.py                        # writes data/predictions/2026-2027_GW<n>.csv
python predict.py --if-starts-tomorrow   # only runs if the next gameweek starts tomorrow
```

Predictions only go one gameweek ahead, because features for later gameweeks depend on results that haven't happened yet. A saved gameweek is never overwritten; delete its file by hand to redo it.

Score predictions and build the page:

```
python score_results.py
python build_site.py                     # writes ../site/index.html
```

Test on this season's data:

```
python holdout_gameweek.py               # hold out the latest finished gameweek
python holdout_gameweek.py --rolling     # every finished gameweek, pooled
```

The GitHub Actions workflow runs the refresh, prediction, scoring and page build every day at 09:17 UTC and publishes the page.
