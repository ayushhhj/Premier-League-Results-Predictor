# Decisions Log

A running log of choices made on this project and the reasoning behind them, kept for future reference.

## 2026-09-17 — Prediction target: match outcome first

Decided to predict match outcome (Home Win / Draw / Away Win) as a classification problem before attempting exact scoreline prediction. Outcome classification is more tractable with Random Forest / XGBoost and gives a solid baseline before tackling the harder scoreline problem later.

## 2026-09-17 — Data source: live API

Decided to pull match data from a live football API rather than a static historical CSV dataset, so the pipeline can also be used to fetch upcoming fixtures for live predictions later (needed for Phase 3's weekly refresh).

Settled on **live-football-api.com** as the provider. Auth is a simple `api_key` query parameter (no header needed). One kink: league IDs are opaque generated strings, not readable slugs — the Premier League's `league_id` is `2kwbbcootiqqgmrzs6o5inle5`, found via the `/leagues` endpoint.

## 2026-09-18 — Data fetching pipeline built (standings, fixtures, h2h)

`src/data_fetch.py` pulls and saves three things to `data/raw/`, one call per season:

- `get_league_standings(season)` → `data/raw/league_standings_<season>.json`
- `get_league_matches(season)` — loops `/league_fixtures` over all 38 gameweeks → `data/raw/league_fixtures_<season>.json`
- `get_h2h_matches(season)` — extracts every match's `id` from that season's saved fixtures file, then calls `/h2h` per match → one file per match under `data/raw/matches/` (only ever called for the current season — see the h2h-scope decision below)

All three take `season` as a parameter (a `SEASONS` constant lists the current + 4 prior seasons) rather than being hardcoded, and filenames are tagged per season so fetching one season doesn't overwrite another's saved data.

Fetching all 38 weeks of one season back-to-back with no delay tripped the API's rate limit partway through (9 of 38 weeks failed with "Too many requests"). Fixed by adding a `time.sleep(0.5)` between requests in the fixtures loop.

Match-specific endpoints (`h2h`, and later `lineups`/injuries if revisited) need a `match_id`, which is only available after fetching fixtures — so fixtures must be pulled before h2h can run. `extract_match_ids()` reads the already-saved fixtures file rather than re-fetching, to avoid burning extra API credits.

## 2026-09-17 — Deferred: lineups, injuries, and other time-sensitive data

Decided to leave out lineups, injuries, and any other data that's only accurate close to kickoff, for now. Two reasons:

- **Availability timing**: lineups are typically only released ~1 hour before kickoff, and injury status can change day-to-day right up until the match. If predictions are meant to be published well ahead of kickoff (e.g. right after the previous matchweek, for the dashboard's weekly refresh), this data simply isn't reliably available yet at prediction time.
- **Data leakage risk**: training a model on injury/lineup data captured close to or after the match (rather than reflecting what was actually knowable at prediction time) would make the model look better in testing than it will perform in real use, since it's implicitly using information from the future relative to when a real prediction would be made.

Suspensions are the exception — since they're derived from accumulated card history, they're fully knowable in advance and don't have this problem, so they're kept in scope.

Revisit lineups/injuries once the actual prediction-timing strategy (how close to kickoff predictions get published) is decided — they become more useful the closer to kickoff predictions are made.

## 2026-09-18 — Training data window: 5 seasons

Decided to pull fixtures (and everything derived from them) from 5 past seasons for the training set — enough matches for a reasonable sample size while staying reasonably close to the current squads/league era, rather than reaching back decades where the league composition and player pool are barely comparable.

## 2026-09-18 — h2h scope and form source

Inspected the actual `/h2h` response and found it bundles four things together: `h2h` (the real head-to-head meetings between the two specific teams, going back decades), `h2h_summary` (win/draw/loss counts aggregated from that list), and `home_form`/`away_form` (each team's last 5 matches in *any* competition — unrelated to the head-to-head matchup itself).

- **Keeping h2h as a feature.** Small per-pairing sample size is a fair caveat, but dominance/stylistic patterns between specific clubs are a real phenomenon, and a Random Forest / XGBoost model can down-weight it on its own if it turns out not to be predictive — no cost to including it.
- **Form feature will come from the standings endpoint's Premier-League-only `form` field, not h2h's `home_form`/`away_form`.** Competition intensity and squad rotation (cup/UCL games are often rotated) make all-competition form a noisy proxy for Premier League form specifically.
- **h2h only needs to be pulled from the current season's fixtures, not from all 5 training seasons.** `/h2h` already returns the full historical record up to today regardless of which match_id triggers it, and every Premier League pairing already meets (home and away) within a single season — so the current season's ~380 matches cover every unique pairing already. Pulling it again per training season would be redundant and cost ~5x the API credits for no new information.

**Note for later (feature engineering):** the `h2h` list returned is "everything up to today," so when building a training row from a past match, that list must be filtered to only meetings dated *before* that match — otherwise a historical row could see head-to-head results that hadn't happened yet at the time.

**Correction to the form source noted above:** the standings endpoint's `form` field is only a single current-day snapshot, so it's only valid for generating live predictions (current season, upcoming fixture). For training rows built from past seasons, form instead has to be computed point-in-time from raw match results (implemented as `compute_form()` in `src/features.py`), since the standings snapshot can't tell you what a team's form looked like partway through a past season.

## 2026-09-18 — Deferred: suspensions

Decided to drop suspension tracking (from accumulated card history) rather than keep it as a planned-but-unimplemented feature. The API only exposes aggregate season card counts, not game-by-game card data — a running total doesn't tell you whether a player is *currently* one yellow away from a ban, which was the actual point of tracking this. Revisit alongside lineups/injuries if a data source with game-by-game card data turns up later.

## 2026-09-18 — Deferred: recency-weighted form

`compute_form()` currently gives equal weight to each of a team's last 5 matches (points won, as a percentage of 15). This treats an improving team ("LLLWW") the same as a collapsing one ("WWLLL"), which loses information about trend. Two candidate approaches to weight more recent matches higher, noted for later:

- **Linear weights** — weight the last N results `1, 2, ..., N` (oldest to newest), so the most recent game counts N× the oldest.
- **Exponential decay** — weight = `decay^games_ago` (e.g. `decay=0.8`), decaying smoothly rather than linearly.

Not implemented yet — shipping the flat-percentage version first and revisiting weighting once the rest of the pipeline works end to end.

## 2026-09-18 — Feature engineering pipeline built

`src/features.py` turns the raw JSON from `data_fetch.py` into model-ready DataFrames:

- `flatten_fixtures(season)` — flattens the nested `weeks → matches` structure, tagging each match with its `week` number
- `build_h2h_lookup(season)` — maps `{team_a, team_b} → h2h data`, built once from the current season's fixtures and reused for that pairing across all training seasons
- `compute_h2h_stats(...)` — recomputes home/away/draw wins fresh from the filtered h2h list rather than trusting the API's own `h2h_summary` field, which is oriented to whichever team was "home" when that data happened to be fetched (could be reversed for a given row)
- `build_row(...)` — assembles one match's feature row (+ `result` label if finished)
- `build_training_set(seasons)` — all finished matches across the given seasons → labeled DataFrame (1,561 rows across the 5 seasons as of this writing)
- `build_prediction_set(season)` — features for the **next unplayed matchweek only**, not the rest of the season — predicting further out would require already knowing the results of the weeks in between, which haven't been played yet either. This is also what makes the weekly-refresh dashboard idea work: predictions regenerate one matchweek at a time as results come in.

Bugs hit while building this, worth remembering the pattern of:
- The `h2h` list uses a combined `"score": "4-0"` string, while fixtures data splits scores into separate `home`/`away` fields — different shapes for what looks like the same kind of data.
- Match `week` numbers come back from the API as **strings** (`"10"` sorts before `"2"` lexicographically) — the same species of bug as comparing goal-score strings, just in a new field. Fixed by casting to `int` at the point of flattening.

## Open questions

- Whether to persist the built training/prediction DataFrames to `data/processed/`, or just rebuild them fresh each time from `data/raw/` (rebuilding all 5 seasons currently takes well under a second, so persistence may not be worth the added complexity — leaning toward rebuilding fresh, not yet finalized).
- Where trained models / prediction logs will live once Phase 3 (web dashboard) starts.
