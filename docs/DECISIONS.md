# Decisions Log

A running log of choices made on this project and the reasoning behind them, kept for future reference.

## 2026-09-17 — Prediction target: match outcome first

Decided to predict match outcome (Home Win / Draw / Away Win) as a classification problem before attempting exact scoreline prediction. Outcome classification is more tractable with Random Forest / XGBoost and gives a solid baseline before tackling the harder scoreline problem later.

## 2026-09-17 — Data source: live API

Decided to pull match data from a live football API rather than a static historical CSV dataset, so the pipeline can also be used to fetch upcoming fixtures for live predictions later (needed for Phase 3's weekly refresh).

Settled on **live-football-api.com** as the provider. Auth is a simple `api_key` query parameter (no header needed). One kink: league IDs are opaque generated strings, not readable slugs — the Premier League's `league_id` is `2kwbbcootiqqgmrzs6o5inle5`, found via the `/leagues` endpoint.

## 2026-09-18 — Data fetching pipeline built (standings, fixtures, h2h)

`src/data_fetch.py` now pulls and saves three things to `data/raw/`:

- `get_league_standings()` — current-season standings (position, points, form, goals for/against) → `data/raw/league_standings.json`
- `get_league_matches()` — loops `/league_fixtures` over all 38 gameweeks of the season → `data/raw/league_fixtures.json`
- `get_h2h_matches()` — extracts every match's `id` from the saved fixtures file, then calls `/h2h` per match → one file per match under `data/raw/matches/`

Fetching all 38 weeks back-to-back with no delay tripped the API's rate limit partway through (9 of 38 weeks failed with "Too many requests"). Fixed by adding a `time.sleep(0.5)` between requests in the fixtures loop.

Match-specific endpoints (`h2h`, and later `lineups`/injuries if revisited) need a `match_id`, which is only available after fetching fixtures — so fixtures must be pulled before h2h can run. `extract_match_ids()` reads the already-saved fixtures file rather than re-fetching, to avoid burning extra API credits.

## 2026-09-17 — Deferred: lineups, injuries, and other time-sensitive data

Decided to leave out lineups, injuries, and any other data that's only accurate close to kickoff, for now. Two reasons:

- **Availability timing**: lineups are typically only released ~1 hour before kickoff, and injury status can change day-to-day right up until the match. If predictions are meant to be published well ahead of kickoff (e.g. right after the previous matchweek, for the dashboard's weekly refresh), this data simply isn't reliably available yet at prediction time.
- **Data leakage risk**: training a model on injury/lineup data captured close to or after the match (rather than reflecting what was actually knowable at prediction time) would make the model look better in testing than it will perform in real use, since it's implicitly using information from the future relative to when a real prediction would be made.

Suspensions are the exception — since they're derived from accumulated card history, they're fully knowable in advance and don't have this problem, so they're kept in scope.

Revisit lineups/injuries once the actual prediction-timing strategy (how close to kickoff predictions get published) is decided — they become more useful the closer to kickoff predictions are made.

## Open questions

- How far back live-football-api.com's data goes, and how much historical data is needed for a reasonable training set.
- Suspension tracking (from accumulated card history via squad data) is planned but not yet implemented.
- Where trained models / prediction logs will live once Phase 3 (web dashboard) starts.
