# Decisions log

A running record of the choices made on this project and the reasons for them.

## 2026-09-17: Predict match outcome first

The first target is the match outcome (home win, draw or away win) as a classification problem, with exact scorelines left for later. Outcome classification is easier to handle with Random Forest or XGBoost and gives a baseline before the harder scoreline problem.

## 2026-09-17: Data source is a live API

Match data comes from a live football API rather than a static historical CSV, so the same pipeline can fetch upcoming fixtures for live predictions later (Phase 3's weekly refresh needs this).

The provider is live-football-api.com. Authentication is an `api_key` query parameter with no header. League IDs are opaque generated strings, not readable slugs: the Premier League's `league_id` is `2kwbbcootiqqgmrzs6o5inle5`, found through the `/leagues` endpoint.

## 2026-09-18: Data fetching pipeline (standings, fixtures, h2h)

`src/data_fetch.py` saves three things to `data/raw/`, one call per season:

- `get_league_standings(season)` writes `data/raw/league_standings_<season>.json`.
- `get_league_matches(season)` loops `/league_fixtures` over all 38 gameweeks and writes `data/raw/league_fixtures_<season>.json`.
- `get_h2h_matches(season)` reads every match `id` from that season's saved fixtures, calls `/h2h` for each and writes one file per match under `data/raw/matches/`. It's only run for the current season (see the h2h scope entry below).

All three take `season` as a parameter (a `SEASONS` constant lists the current season and the four before it), and filenames include the season so one season's fetch doesn't overwrite another's.

Fetching all 38 weeks of a season back to back hit the API's rate limit: 9 of 38 weeks failed with "Too many requests". A `time.sleep(0.5)` between requests in the fixtures loop fixed it.

Match-specific endpoints (`h2h`, and `lineups` or injuries if they come back into scope) need a `match_id`, which only exists after fixtures are fetched, so fixtures have to come first. `extract_match_ids()` reads the saved fixtures file instead of fetching again, to save API credits.

## 2026-09-17: Deferred lineups, injuries and other late data

Lineups, injuries and anything else that's only accurate close to kickoff are left out for now, for two reasons:

- Timing: lineups usually come out about an hour before kickoff, and injury news can change daily until the match. If predictions go out well before kickoff (for example right after the previous gameweek, for the dashboard), this data won't reliably exist yet.
- Leakage: training on injury or lineup data captured close to or after the match, rather than what was known at prediction time, would make the model look better in testing than it will be in real use.

Suspensions were the exception, since they follow from card history and are known in advance. (They were dropped later; see the suspensions entry below.)

Lineups and injuries are worth revisiting once it's decided how close to kickoff predictions get published. The closer to kickoff, the more useful they are.

## 2026-09-18: Training window of 5 seasons

Fixtures and everything derived from them come from 5 seasons. That's enough matches for a reasonable sample while staying close to current squads, rather than going back decades to a league with a very different makeup.

## 2026-09-18: h2h scope and form source

The `/h2h` response bundles four things: `h2h` (the actual meetings between the two teams, going back decades), `h2h_summary` (win, draw and loss counts from that list), and `home_form`/`away_form` (each team's last 5 matches in any competition, unrelated to the head-to-head).

- h2h stays as a feature. Each pairing has a small sample, but some clubs do have a consistent edge over others, and Random Forest or XGBoost can down-weight it if it turns out not to help.
- Form will come from the standings endpoint's Premier League `form` field, not from h2h's `home_form`/`away_form`. Cup and European matches are often played with rotated squads, which makes all-competition form a noisy stand-in for league form.
- h2h only needs fetching for the current season's fixtures, not all 5 seasons. `/h2h` returns the full history whichever match_id requests it, and every pairing meets home and away within one season, so the current season's ~380 matches cover every pairing. Fetching per season would cost about 5 times the API credits for no new information.

Note for feature engineering: the `h2h` list runs up to today, so for a past match it has to be filtered to meetings before that match. Otherwise a historical row would see results that hadn't happened yet.

Correction to the form source above: the standings `form` field is a single snapshot of today, so it only works for live predictions of upcoming fixtures. Training rows from past seasons need form calculated point-in-time from match results, which is what `compute_form()` in `src/features.py` does.

## 2026-09-18: Deferred suspensions

Suspension tracking from card history is dropped rather than kept as a planned feature. The API only gives total season card counts, not cards per game, and a running total can't say whether a player is one yellow card away from a ban, which was the point. Worth revisiting with lineups and injuries if a source with per-game card data turns up.

## 2026-09-19: Recency-weighted form (exponential decay)

Revisited the flat form calculation (entry below) once a trained model gave a baseline to compare against. Chose exponential decay over linear weights. `compute_form()` now weights the i-th most recent match (i=0 is the latest) by `decay**i` with `decay=0.8`, and divides by the weights actually used rather than a fixed 15. That also fixes an older problem: a team with fewer than 5 earlier matches (week 1 of a season, say) had its form understated, because the old version always divided by 15.

Also added home-only and away-only versions of form and goal stats (a `venue="home"`/`"away"` parameter on `team_matches_before`, `compute_form` and `compute_goal_stats`). Overall form can hide a real home/away difference, and this needed no new data, only filtering of matches already fetched.

Combined effect on the 2025/2026 test season (trained on 2022/2023 to 2024/2025): Random Forest went from 40.3%, below the majority-class baseline of 42.6%, to 46.3%, ahead of the baseline for the first time, and its macro F1 went from 0.36 to 0.39. XGBoost stayed near the baseline (39.7% to 42.1%): the venue splits alone made it worse, and weighted form brought it back level but not ahead.

## 2026-09-18: Deferred recency-weighted form (superseded by the entry above)

`compute_form()` used to weight each of a team's last 5 matches equally (points won as a share of 15). That treats an improving team ("LLLWW") the same as one in decline ("WWLLL"). Two options were considered:

- Linear weights: weight the last N results 1, 2, ..., N from oldest to newest, so the latest counts N times as much as the oldest.
- Exponential decay: weight = `decay^games_ago` (for example `decay=0.8`), which falls off smoothly. This is the one that was implemented (see above).

## 2026-09-18: Feature engineering pipeline

`src/features.py` turns the raw JSON from `data_fetch.py` into DataFrames for the models:

- `flatten_fixtures(season)` flattens the nested weeks and matches, tagging each match with its `week`.
- `build_h2h_lookup(season)` maps each pair of teams to its h2h data, built once from the current season's fixtures and reused for that pairing in every season.
- `compute_h2h_stats(...)` counts home wins, away wins and draws from the filtered h2h list instead of using the API's `h2h_summary`, which is oriented to whichever team was at home in the fixture the data was fetched for, so it can be the wrong way round for a given row.
- `build_row(...)` builds one match's feature row, plus the `result` label if it's finished.
- `build_training_set(seasons)` builds labelled rows for every finished match in the given seasons (1,561 rows across the 5 seasons at the time of writing).
- `build_prediction_set(season)` builds features for the next unplayed gameweek only. Going further ahead would need results from the gameweeks in between, which haven't been played. This is also what makes the weekly dashboard work: predictions are made one gameweek at a time as results come in.

Bugs found along the way:
- The `h2h` list stores the score as one string (`"score": "4-0"`), while fixtures split it into `home` and `away` fields.
- The API returns `week` numbers as strings, so `"10"` sorts before `"2"`. This is the same kind of bug as comparing goal scores as strings. Fixed by converting to `int` when flattening.

## 2026-09-20: Rest days

Added `home_rest_days`/`away_rest_days`: days since each team's last competitive match, in any competition, before the fixture date. This needed a new source. `get_team_matches(team_id, season)` in `data_fetch.py` calls `/team_matches`, which covers a team's matches in every competition, unlike `league_fixtures`. Cup and European matches affect rest even though they don't appear in Premier League data. Fetched for all ~20 teams in all 5 seasons (~100 calls) with `get_all_team_matches()`, run by hand like h2h rather than from `data_fetch.py`'s `__main__`.

Preseason "Club Friendly" matches are excluded (`EXCLUDED_COMPETITIONS`). They're heavily rotated and don't cause real fatigue, and counting them would understate rest before a season's first real matches. When a team has no earlier competitive match (its first competitive fixture of the season, for example), `compute_rest_days` returns `7`, a normal one-match-a-week gap, rather than a blank.

Result on the 2025/2026 test season: Random Forest went from 46.3% to 46.8%, and XGBoost from 42.1% to 44.5%, its first time clearly beating the baseline (42.6%).

## 2026-09-20: Tried Dixon-Coles (not adopted, kept for later)

Tried the Dixon-Coles adjusted Poisson model (from the `penaltyblog` package, added to `requirements.txt`; it brings in about 57 dependencies, most unrelated to this model) as a football-specific alternative to the general classifiers. Unlike the other models, it works from team identity and past goals (`home_score`/`away_score` were added to `build_row`'s output for it), estimating each team's attack and defence strength by maximum likelihood.

Findings:
- It can't predict for a team that isn't in the training data. A newly promoted club with no Premier League matches in the training seasons has no fitted parameters. That affected 38 of 380 rows (10%) of the 2025/2026 test season. `evaluate_dixon_coles` skips these and reports how many, to keep comparisons fair.
- On the 342 rows it could predict, Dixon-Coles scored 47.7%, against 50.0% for ordered logistic regression on the same rows.
- An ensemble of Dixon-Coles and Ordered Logit (averaging their H/D/A probabilities, after reordering, since Dixon-Coles returns [H, D, A] and Ordered Logit [A, D, H]) scored 47.9%, worse than Ordered Logit alone. Averaging in a weaker model dilutes the stronger one, as with the earlier RF and XGBoost ensemble.

Not adopted for Phase 1. Kept in the code because it's the only model here that predicts goal rates, which makes it the obvious starting point for Phase 2 (scorelines).

## 2026-09-20: Best model so far is ordered logistic regression

Added `train_ordered_logit`/`evaluate_ordered_logit` to `src/train.py`, using `statsmodels.miscmodels.ordinal_model.OrderedModel` (a new dependency, added to `requirements.txt`). RF and XGBoost treat H, D and A as unrelated classes. This model puts them on one ordered scale, away win < draw < home win, roughly "how much did the home team win by", which tree models can't take advantage of. It needs its own label mapping, `ORDINAL_LABEL_MAP = {"A": 0, "D": 1, "H": 2}`, separate from the arbitrary `LABEL_MAP` the other models use, because here the order matters.

Result on the 2025/2026 test season: 49.2%, the best so far (the previous best was tuned Random Forest at 47.6%).

It has the same weakness as every other high-accuracy result: draws got 0.00 precision, recall and F1. All 104 actual draws were predicted as home or away wins. This is the fourth model or technique (tuned RF, tuned XGBoost, the rejected draw-threshold experiment, and this one) to end up never predicting a draw when maximising accuracy, which points to the feature set rather than any one model. Since accuracy was the stated priority, this stands as the best model, with that tradeoff accepted.

## 2026-09-20: Rejected opponent-strength adjustment

Added `home_opponent_strength`/`away_opponent_strength`: the average goal difference of the opponents in a team's last 5 matches, each measured as of that match's date. The motivation: `form`, `points` and `goal_diff` count a win over a relegation side the same as a win over a title contender. (This came out of a wider look at confounders and correlated features in the feature set.)

On the 2025/2026 test season all three models got worse: Random Forest 47.6% to 46.1%, XGBoost 47.1% to 46.6%, and the ensemble 47.6% to 46.6%. The walk-forward CV score during tuning was slightly better (0.5171 against 0.5145 for RF), so CV and the test season disagreed. Feature importance put the new features in the lower middle for both models, near rest days and h2h.

Likely cause: it's a point-in-time stat (the opponents' goal difference) built on top of another point-in-time stat, often measured early in a season when there's little data, so the noise compounds. Opponent strength is a well-known confound in sports analytics, but with about 1,140 training rows there wasn't enough signal. Reverted. Worth another look if the training set grows a lot, or with a steadier measure of opponent strength (a full-season average, for example, rather than an early-season snapshot).

## 2026-09-20: Rejected raising the bar for predicting draws

Tried predicting a draw only when P(draw) cleared a higher threshold (0.4 or 0.5) instead of taking the most likely outcome, on the idea that draws are the hardest to call and de-emphasising them might raise overall accuracy. On the 2025/2026 test season, Random Forest went 46.8% (default), 46.1% (0.4) and 47.4% (0.5); XGBoost went 44.5%, 44.2% and 43.9%. The one improvement (RF at 0.5) was under 1 point, while its correct draws fell from 8 of 104 to 1 of 104. That's a real cost for a gain within the noise of a 380-match test set. Reverted.

## 2026-09-20: Deferred pre-match odds (api-football.com)

Considered pre-match betting odds from api-football.com as a feature. Odds are one of the strongest known predictors of results, because they include information like team news and market sentiment that stats can't capture. The free tier does include odds (100 calls a day), but it wasn't clear whether the odds endpoint keeps historical odds for past matches or only covers upcoming fixtures. The docs didn't say, and fetching them directly returned a 403.

That matters: without historical odds this could only be used for live predictions (like lineups and injuries), not for training. Even with historical odds, 100 calls a day against ~1,900 past matches would take about 19 days, unless there's a bulk endpoint by date or league, which also wasn't confirmed. api-football.com also uses its own team and match IDs, so joining its odds to existing fixtures would mean matching on team name and date.

Not pursued for now. Worth revisiting if historical odds and bulk fetching are confirmed.

## 2026-09-21: Ruled out squad market value

Considered squad market value as a feature to capture transfer-window changes. Everything in the feature set reacts to matches already played, so a squad that just spent heavily doesn't look stronger until its results catch up weeks later.

Tested both relevant endpoints directly, since the docs say nothing:
- `/team_squad` with a `season` parameter returns historically accurate squads: Arsenal's 2022/23 squad has Ramsdale in goal, and 2025/26 has David Raya, Declan Rice, Kai Havertz and other real transfers, not today's squad relabelled. But it has no `market_value`.
- `/player` has `market_value`, but only today's figure, with no date or season parameter. Using it for a past training row would leak information, like the standings `form` field: today's value reflects everything since, including the matches being predicted.

Ruled out as a training feature. It could work for live predictions only, but that's narrower than intended and not worth about 2,500 calls (`/team_squad`, then `/player` for each player: ~25 players, 20 teams, 5 seasons). This is the third time this API's player or squad data has turned out to be today's snapshot rather than historical. Lineups' coach data is historical; standings, injuries and market value aren't.

## 2026-09-21: Looked at the `soccerdata` package

Checked `soccerdata`, a Python library that wraps several football data sources (FBref, Understat, WhoScored, ClubElo, Football-Data.co.uk, Sofascore, ESPN, SoFIFA), looking for metrics about how teams play rather than just goals. Tested the two most relevant sources directly:

- Understat works and needs no browser scraping. It returns per-match `xg`/`np_xg` (expected goals and non-penalty expected goals), `ppda` (passes per defensive action, a measure of pressing) and `deep_completions` (completed passes into dangerous areas). These measure underlying performance rather than repackaging goals. Coverage runs from 2020/21 to 2025/26, enough for the training window.
- ClubElo is unreliable. `api.clubelo.com` returned a 502, checked twice against the raw URL without the library. It's a small project run by one person with no uptime guarantee, so depending on it is an ongoing risk, not a one-off outage.
- The package brings in about 45 dependencies, including `selenium`/`seleniumbase`, because some of its sources (WhoScored) need browser automation. Understat doesn't, but installing the package brings them in anyway.

Using Understat's xG needs two pieces of work first:
1. Team names differ from live-football-api.com's (for example "Bournemouth" on Understat against abbreviated forms elsewhere), so it needs a name-to-team_id mapping for the ~20 teams.
2. A match's xG only exists after it's played, so using it to predict that same match would leak the answer, even more directly than injuries or lineups would. It has to be a rolling point-in-time feature (average xG over a team's last N matches before the date), like `compute_form`/`compute_goal_stats`, not a column joined in directly.

## 2026-09-21: Deferred pre-match odds by choice

Revisited odds now that `soccerdata`'s Football-Data.co.uk source looks like it solves the historical-data problem (that site is known for clean historical odds alongside results). Still not pursuing them, for a different reason: odds are almost certainly the strongest single feature available, since bookmakers use information (team news, sharp money) that stats can't replicate. Using them would mean much of any accuracy gain came from repackaging the market's own prediction rather than anything built here. Worth trying later as a ceiling check ("how much do our features capture compared with what the market knows"), not as a shortcut now.

## 2026-09-21: Dropped deep completions; best result so far

Dropped `home_deep_completions`/`away_deep_completions` and kept `home_xg`/`away_xg`, the last correlated pair (r=0.739/0.748), the same treatment `home_points`/`home_goal_diff` got earlier. xG was kept because it's the more standard metric and more directly tied to results; deep completions is more of a build-up indicator.

After this, the correlation check finds no correlated pairs in the 13-feature set. Accuracy improved, mostly for the linear models: Random Forest 47.9% to 48.7%, Ordered Logit 48.9% to 51.1%, and Lasso 49.2% to 51.8%, the best of the project so far. That fits the earlier look at correlated features: Lasso and Ordered Logit are the models most affected by correlated inputs, and they improved most once the last redundant pair was gone. Tuned Random Forest went the other way (48.4% to 47.1%), but its walk-forward CV score also fell (0.5263 to 0.5145) and tuning picked a different, less regularised set of hyperparameters, so that looks like tuning variance on the new feature set rather than a result against the pruning.

## 2026-09-21: Dixon-Coles time decay; season-weighted Random Forest not adopted; XGBoost and ensembles commented out

Added time-decay weighting to `train_dixon_coles`. The `DixonColesGoalModel` constructor takes an optional `weights` array that wasn't being used, so every match counted equally regardless of age. Tried half-lives of 730, 365, 180, 90 and 45 days; 180 days was best, going from 47.66% to 47.95% on the 342 comparable test rows. A small improvement that doesn't close the gap to the leading models.

Finished `compute_season_weights` (the same decay idea as `compute_form`, applied per season) and added `sample_weight` to `train_random_forest`. The result was mixed to negative, as with the Dixon-Coles weighting. Untuned RF accuracy dropped slightly (47.9% to 47.1%) though draw precision improved (0.30 to 0.44). Tuned RF got worse on every measure: accuracy fell from 48.4% to 45.5%, and draw recall fell from 0.02 to 0. Tuning already trades draw detection for accuracy, and weighting pushed further in the same direction. Not adopted: `__main__` went back to plain unweighted `train_random_forest(train_df)` calls, but `compute_season_weights` and the `sample_weight` parameter stay as tested code in case they're worth revisiting.

Also commented out (not deleted) XGBoost (the import, `train_xgboost`, `XGB_PARAM_GRID` and its uses in `__main__`) and both ensemble functions (RF plus XGBoost averaging, and Dixon-Coles plus Ordered Logit). XGBoost has trailed Random Forest, Ordered Logit and Lasso in every experiment, and neither ensemble ever beat its best member.

## 2026-09-21: Added xG, PPDA and deep completions; dropped PPDA

Added `home_xg`/`away_xg`, `home_ppda`/`away_ppda` and `home_deep_completions`/`away_deep_completions` as rolling point-in-time features (through `compute_xg_form`, which works like `compute_form`), from Understat via the new `understat_fetch.py` and a team-name map (`UNDERSTAT_TEAM_MAP`, 27 teams, built by comparing the two sources' name lists).

On the 2025/2026 test season: Random Forest 48.2% to 48.7%, XGBoost 46.8% to 45.5% (worse), Ordered Logit 49.2% to 49.7%. Mixed on accuracy, like most additions. Feature importance was clearer: deep completions and xG ranked high for both models (`home_deep_completions` was RF's second most important feature, `away_xg` was XGBoost's second). PPDA ranked in the bottom third for both, where h2h and rest days usually land.

Also found a new correlated pair: `home_xg` and `home_deep_completions` (0.739), and the away equivalent (0.748). That makes sense, since teams completing more passes into dangerous areas create better chances.

Dropped `ppda` for its consistently weak showing, as with `home_points` earlier, removing it from `build_row` and `FEATURE_COLUMNS`.

Considered, then rejected, a bigger change: replacing the goal-difference features with xG, on the theory that one is redundant with the other. The actual correlation between `home_xg` and `home_goal_diff` is only 0.527, well below the 0.7+ used elsewhere in this project to call features redundant (the points/venue_points and goal_diff/venue_goal_diff pairs were 0.87 to 0.98). xG measures the quality of chances created; goal difference measures what actually happened, including finishing and luck. They're related but different, and `home_goal_diff` has been the most consistently important feature in every model so far. Replacing it would have swapped a proven signal for a moderately correlated one. Not done.

Did a smaller version instead: dropped `home_goal_diff`/`away_goal_diff` and kept `home_venue_goal_diff`/`away_venue_goal_diff`. That pair is redundant (r=0.906/0.865, rechecked on current data), like the points pair. On the 2025/2026 test season: RF 48.7% to 48.4%, XGBoost 45.5% to 46.8%, Ordered Logit 49.7% to 48.9%. Small mixed changes, within noise. Two correlated pairs remain (both xG with deep completions), down from four, and `home_xg`/`home_deep_completions` are now the top-ranked features for both RF and XGBoost.

## 2026-09-21: Odds ceiling check, and why the first single-season result misled

Ran the ceiling check planned in the "Deferred pre-match odds by choice" entry. Fetched historical 1X2 odds from Football-Data.co.uk through `soccerdata`'s `MatchHistory` for all 4 completed seasons (new file: `src/odds_fetch.py`) and joined them in as `market_prob_home`/`market_prob_draw`/`market_prob_away` (implied probabilities with the bookmaker margin removed, normalised to sum to 1) in a new odds section of `src/features.py`. Kept as a separate experiment: not added to `FEATURE_COLUMNS` or the main pipeline.

The first test, on the usual 2025/2026 test season only, was surprising: all three models got worse with odds (RF -3.2pp, Lasso -3.6pp, Ordered Logit -2.7pp). Since odds are one of the best-established predictors in football, that looked wrong. Checked data quality first (team names matched on all 1,520 rows, no join bug) and correlation with existing features (moderate, 0.3 to 0.5, not severe), then reran the comparison walk-forward across all 3 train/test splits instead of one season, for Random Forest and Lasso:

| Test season | RF without odds | RF with odds | Lasso without odds | Lasso with odds |
|---|---|---|---|---|
| 2023/2024 | 49.7% | 52.9% (+3.2pp) | 55.3% | 57.9% (+2.6pp) |
| 2024/2025 | 47.9% | 53.4% (+5.5pp) | 49.5% | 52.1% (+2.6pp) |
| 2025/2026 | 48.7% | 45.5% (-3.2pp) | 51.8% | 48.2% (-3.7pp) |

Odds help a lot on 2 of the 3 test seasons (+2.6 to +5.5pp) for both models and only hurt on 2025/2026, the season used as the test set for almost every other comparison in this project. The first result came from that one season, not from the feature. Averaged over all 3, odds add about 1.5 to 1.8 points, in line with the usual finding that the market is hard to beat.

This has a wider implication, noted but not acted on: every other "X helped" or "X didn't help" call so far was judged on that same 2025/2026 season, and this shows it can flip a conclusion. Earlier experiments weren't redone, but any past "no effect" result is worth rechecking if it comes up again.

Odds stay out of the main feature set. This was a ceiling check ("how much of the accuracy would just be the market's own prediction"), not a decision to adopt them, for the reasons in the earlier deferred-odds entry.

## 2026-09-22: Fixed missing current-season odds; season weighting helps once odds are included

Two follow-ups from the odds ceiling check.

Missing data, found and fixed: testing odds on the current season (2026/2027) showed every row falling back to the neutral `market_prob_*` default (0.333 each). `odds_fetch.py` had only been run for the 4 completed seasons, so `build_odds_lookup` found nothing and used the fallback without any warning. Fetched it (`get_odds('2026/2027')`). That exposed a second gap: `ODDS_TEAM_MAP` was missing `Hull` and `Coventry`, this season's two promoted teams. They'd been added to `UNDERSTAT_TEAM_MAP` but not to the odds map, so 9 of 50 rows still fell back to the default after the fetch. Added both IDs, copied from `UNDERSTAT_TEAM_MAP`, which uses the same ID scheme. All 50 matches played this season now have real odds.

After the fix, trained on everything up to 2025/2026 and tested on 2026/2027 so far (50 matches). Both RF and Lasso improved with odds (RF 42.0% to 44.0%, Lasso 44.0% to 48.0%). That makes 3 of 4 checks agree (2023/2024, 2024/2025 and this one), with 2025/2026 still the outlier.

Season-decay weighting, retested with odds as a feature, now gives the opposite result. The 2026-09-21 entry rejected `compute_season_weights` for Random Forest because it hurt without odds. Retried several decay values with `market_prob_home`/`market_prob_draw` in the feature set (still only in one-off comparison scripts, not `FEATURE_COLUMNS`) on the 2025/2026 test season:

| Decay | RF accuracy | RF draw precision/recall | RF macro-F1 | Lasso accuracy | Lasso draw precision/recall | Lasso macro-F1 |
|---|---|---|---|---|---|---|
| 1.0 (unweighted) | 45.5% | 0.14 / 0.03 | 0.359 | 48.2% | 0.00 / 0.00 | 0.360 |
| 0.9 | 46.6% | 0.17 / 0.04 | 0.372 | 48.2% | 0.33 / 0.01 | 0.366 |
| **0.8** | **48.4%** | 0.33 / 0.05 | 0.387 | 48.4% | 0.60 / 0.03 | 0.377 |
| 0.7 | 46.6% | 0.24 / 0.06 | 0.379 | **49.2%** | 0.67 / 0.04 | 0.388 |
| 0.6 | 46.1% | 0.26 / 0.06 | 0.371 | 48.9% | 0.62 / 0.05 | 0.391 |
| 0.5 | 47.4% | 0.24 / 0.05 | 0.382 | 48.4% | 0.67 / 0.08 | **0.400** |

With odds included, weighting helps both models on every measure at every decay value tried, the reverse of the result without odds. The best accuracy was RF at decay 0.8 (48.4%, against 45.5% unweighted) and Lasso at decay 0.7 (49.2%, against 48.2%). Unweighted Lasso never predicts a draw (0.00/0.00, the same draw blindness seen elsewhere). Any weighting changes that, and Lasso's macro-F1 and draw recall keep improving down to decay 0.5, past the point where accuracy peaks. So there's a tradeoff between accuracy and balance, depending on which matters more.

Working theory: odds already measure each team's current strength, since the market prices every match fresh. Before odds, down-weighting older seasons worked against the other features (form, venue stats), which were the model's only measure of how good a team is now. With odds supplying that directly, down-weighting stale rows stops duplicating it and starts helping.

Not adopted. It depends on the same question as the ceiling check (no confirmed source of odds for unplayed fixtures), and weighting was only tested together with odds, not on its own.

## 2026-09-22: Investigated, then shelved, FBref match stats (soccerdata bug)

Looked at `soccerdata`'s FBref reader as a free source of new information beyond Understat and odds: `read_team_match_stats(stat_type=...)` gives per-match `shooting` (shots, shots on target, SoT%, goals per shot) and `misc` (cards, fouls, offsides, crosses, interceptions, tackles won, aerial stats) tables. Both are per match and unlike anything in the feature set, and `misc` would be the project's first defensive data (everything so far measures attacking play).

Testing turned up two problems:
- Slow, inconsistent scraping. FBref needs headless Chrome (Selenium), unlike the simple HTTP requests for Understat and Football-Data.co.uk, which is a heavier and more fragile dependency (it downloaded a chromedriver on first use). For the same team and season, `shooting` came back in seconds and `misc` took over 2 minutes.
- A bug in `soccerdata` itself that can't be fixed from here. `read_team_match_stats(team=None)` (all teams in one call) and `read_team_match_stats(team=<list of names>)` (the attempted workaround) both fail to attach the right team. With `team=None`, every row came back with `team=NaN` (760 of 760 rows, on a fresh pull), and passing a list crashes inside the library (`soccerdata/fbref.py:451`, `ValueError: Length of values (20) does not match length of index (58)`). Only fetching one team at a time (`team="Arsenal"`) returns correctly labelled data.

So usable data means looping over all ~20 teams one at a time. With the slow and uneven call times, ~20 teams × 4-5 seasons × 2 stat types (150 to 200 scrapes) could take from tens of minutes to a few hours, with a real chance of being rate-limited or blocked partway (FBref is known to push back against scrapers).

Shelved, not ruled out, like ClubElo earlier: the data is fine, but the source is unreliable. The scaffolded `fbref_fetch.py` and a `features.py` section (`FBREF_TEAM_MAP`, `load_fbref_matches`, `fbref_team_matches_before`, `compute_fbref_form`) stay in place, with `fbref_fetch.py`'s fetch-and-save code working for a single team. Worth revisiting if a more reliable way into FBref turns up, or if the per-team loop becomes worth the time and risk.

Also checked and ruled out a shortcut: feeding these stats into Dixon-Coles for scorelines. Every goal model in `penaltyblog` (`DixonColesGoalModel`, `PoissonGoalsModel`, `NegativeBinomialGoalModel`, `BivariatePoissonGoalModel`, `ZeroInflatedPoissonGoalsModel`) takes the same fixed arguments (`goals_home, goals_away, teams_home, teams_away, weights, neutral_venue`), and none accept extra features. Using engineered features in a goals-based scoreline model would mean writing a custom Poisson regression from scratch. Not pursued for now, since scorelines aren't the priority and the current models only predict H/D/A.

## 2026-09-22: Custom Elo rating feature

Wrote an Elo implementation in `features.py` (`expected_score`, `elo_result_value`, `all_matches_chronological`, `compute_elo_lookup`, `get_match_elo`) instead of using ClubElo, which was already shelved as unreliable. Its advantage over every other added source: Elo needs no external data. It's calculated from match results already in `data/raw/`, so the scraping, rate-limit and availability problems that hit Understat, FBref and odds don't apply.

Warm-up seasons matter a lot. Without extra history, every team starts the 2022/23 training window at the initial 1500 rating, Man City and newly promoted sides alike. Fetched one extra season, 2021/2022, only to seed ratings before the training window starts (in `data_fetch.py`, commented out in `__main__` after one run, like `get_all_team_matches`). Not 2019/20 or 2020/21: those were affected by COVID (few or no crowds, which distorted home advantage), and 2021/22 was the first normal season back. 2021/22 isn't in `SEASONS`/`COMPLETED_SEASONS_CHRONOLOGICAL` and never becomes training rows; it's only passed as extra history to `compute_elo_lookup(["2021/2022"] + seasons)`. The warm-up makes a real difference: teams in 2021/22 enter the training window with spread-out ratings (Liverpool +175.7 against no warm-up, Man City +166.4, Everton -61.1), while teams promoted into 2022/23 with no earlier top-flight history stay at 1500, as they should.

A bug caught by testing prediction rows, not only training rows: at first, every prediction row from `build_prediction_set` showed 1500/1500 for every team, including established sides like Arsenal. The cause: `compute_elo_lookup`'s `lookup` dict only has entries for finished matches, so an unplayed fixture's `match_id` was never in it, and `get_match_elo` fell back to the neutral default, which is wrong for a team with real history. Fixed by having `compute_elo_lookup` return two things: the match_id-keyed `lookup` (for training rows, each team's rating before that match) and a new `ratings` dict (each team's latest rating, for prediction rows). `build_row` now checks `match["status"]["status"]` to choose between them. It's the same kind of bug this project has hit before (comparing against the wrong reference point), and it only showed up because prediction rows were tested directly.

Walk-forward result, base 13 features against the same plus `home_elo`/`away_elo`:

| Test season | RF base | RF+elo | diff | Lasso base | Lasso+elo | diff |
|---|---|---|---|---|---|---|
| 2023/2024 | 49.7% | 51.6% | +1.8pp | 55.3% | 57.9% | +2.6pp |
| 2024/2025 | 47.9% | 47.1% | -0.8pp | 49.5% | 51.8% | +2.4pp |
| 2025/2026 | 48.7% | 48.4% | -0.3pp | 51.8% | 47.9% | -4.0pp |

Same pattern as the odds result. For Random Forest it nets out near zero, which looks like noise. Lasso gains on 2 of 3 seasons (+2.4 to +2.6pp) and only drops on 2025/2026, the third finding (after odds and season weighting) where that season is the outlier and the other two agree. Elo and odds both stand in for current team strength, so it's no surprise they behave alike. Not added to `FEATURE_COLUMNS`; it's a one-off comparison, like odds.

## 2026-09-24: 2021/2022 as a training season, adopted for Random Forest only

Tested 2021/22 as training data, not just Elo warm-up. That needed two more sources Elo didn't: Understat xG (`get_understat_matches('2021/2022')` in `understat_fetch.py`) and team matches for rest days (`get_all_team_matches(["2021/2022"])` in `data_fetch.py`, commented out in `__main__` after one run, as before).

A bug in the Understat fetch, outside this project's code: `_understat_season` in `understat_fetch.py` does `season.split('/')[0]`, giving `"2021"`. `soccerdata` treats that as ambiguous and quietly resolved it to 2020/21, the COVID season that choosing 2021/22 was meant to avoid, with only a printed warning. Caught by checking the saved file's dates (September 2020 to May 2021, not August 2021 to May 2022) rather than trusting the fetch. Refetched with `"2021-2022"`, which resolves correctly. Checked that the bug didn't affect the 4 seasons already fetched (2022/23 to 2025/26); all have the right date ranges. The bare-year format in `_understat_season` is still a latent bug if the project ever goes further back.

Walk-forward result, base 13 features, with and without 2021/22 in training:

| Test season | RF no-21/22 | RF +21/22 unweighted | RF +21/22 weighted (0.8) | Lasso no-21/22 | Lasso +21/22 unweighted | Lasso +21/22 weighted (0.8) |
|---|---|---|---|---|---|---|
| 2023/2024 | 49.7% | 52.1% (+2.4pp) | 51.1% (+1.3pp) | 55.3% | 53.7% (-1.6pp) | 54.2% (-1.1pp) |
| 2024/2025 | 48.2% | 51.8% (+3.7pp) | 48.2% (+0.0pp) | 49.5% | 49.5% (+0.0pp) | 50.0% (+0.5pp) |
| 2025/2026 | 44.7% | 46.1% (+1.3pp) | 46.1% (+1.3pp) | 51.8% | 48.7% (-3.2pp) | 51.1% (-0.8pp) |

Unlike odds, Elo and season weighting, this result holds on all 3 seasons, with no 2025/2026 outlier. Random Forest gains on every season (about +2.5pp on average, unweighted), and weighting shrinks the gain. Lasso is flat or worse on every season unweighted, and weighting only partly reduces the loss.

Decision: use 2021/2022 as a training season for Random Forest only, unweighted, and not for Lasso. Random Forest is flexible enough to handle a season from a slightly different era in its own part of the trees without it affecting predictions elsewhere. Lasso fits one linear boundary across all the training data, so any drift between 2021/22 and later seasons just adds conflicting signal. Not yet wired into `train.py`'s `__main__`, since RF would need a different season list from the other models, which means restructuring. Recorded as a decision; wiring it in is a separate step.

Follow-up, not adopted: rather than the blanket `compute_season_weights` decay (which also down-weights 2022/23, a season that has nothing to do with the doubts about 2021/22), tried giving only 2021/22 a lower weight and leaving the rest at full weight. Averaged over the 3 seasons this beat full weight slightly (weight 0.5 or 0.25 gave about 49.3% to 49.4%, against about 48.5% at weight 1.0, roughly +0.9pp). But no single weight won on every season; each preferred a different value. With a small gain and no clear best value, it wasn't worth pursuing. RF with 2021/22 stays plain and unweighted rather than adding another untuned parameter.

## 2026-09-24: Why season weighting still isn't the default

Season weighting has now been tested three times with three different results: harmful for Random Forest without odds (2026-09-21), helpful for RF and Lasso with odds (2026-09-22), and worse than unweighted for RF with 2021/22 added (the entry above). Same technique, same `decay=0.8` starting point, and the outcome depended on what else was in the features or training data at the time.

That inconsistency is why it isn't on by default. There's no evidence yet of a weighting policy that helps regardless of context; every test so far has depended on some other change at the same time (odds, an extra season, the choice of model). Turning it on everywhere because it looked good in its latest context would mean generalising from one good result when this project has already shown it doesn't generalise. `compute_season_weights` and the `sample_weight` plumbing stay as tested code, to use deliberately in an experiment rather than by default.

## 2026-11-03: First out-of-sample test on the current season

Added `src/holdout_gameweek.py`. It holds out a gameweek of 2026/2027, trains on every completed season plus this season's earlier gameweeks, and scores each model on the held-out matches. It reports accuracy, log loss and ranked probability score (RPS, the usual metric for ordered football outcomes), with bookmaker odds (margin removed) shown as a benchmark, not a model input. `--rolling` repeats this for every finished gameweek (train on gameweeks before k, test on gameweek k) and pools the results, since one gameweek is only 10 matches.

The data on disk only went up to GW5 (fetched in late September), so GW5 was the latest gameweek to hold out. The data couldn't be refreshed at the time (no network access to the APIs). Rerun after fetching fresh fixtures, team matches, Understat and odds to test the newest gameweek.

GW5 alone (10 matches): Lasso and Ordered Logit 6 of 10, Random Forest 5, always-home baseline 5, Dixon-Coles 4, bookmakers 4. It was a week of upsets (Brighton 3-0 Arsenal, Brentford 3-0 Chelsea, Spurs 2-3 Villa), so the bookmakers' 4 of 10 shows how noisy a single gameweek is, not that the model beats the market.

Rolling GW1 to GW5 (50 matches, pooled):

| Model | Correct | Accuracy | Log loss | RPS |
|---|---|---|---|---|
| Always home (baseline) | 18/50 | 36% | 4.87 | 0.479 |
| Random Forest | 21/50 | 42% | 1.113 | 0.217 |
| Lasso | 25/50 | **50%** | 1.074 | 0.212 |
| Ordered Logit | 25/50 | **50%** | 1.072 | 0.212 |
| Dixon-Coles | 24/50 | 48% | 1.094 | **0.204** |
| Bookmakers (reference) | 22/50 | 44% | **1.052** | 0.206 |

Lasso and Ordered Logit hold up on unseen current-season data at the same ~50% as on 2025/2026, so there's no sign that leakage inflated the earlier numbers. But with 50 matches the standard error on accuracy is about ±7 points, so the 3-match gap to the bookmakers is noise. On the probability measures, which make the fairer comparison, the bookmakers have the best log loss and Dixon-Coles the best RPS. Only RF predicted any draws, while 16 of the 50 results were draws (32%, against about 24% historically).

Two other findings:
- Gameweek 1 has almost nothing to go on. Every form, venue and xG feature resets to 0 at the start of a season (xG and form were 0 on every GW1 row), so GW1 predictions rest almost entirely on h2h and rest days. RF handles the all-zero pattern badly, predicting 6 draws in GW1 and getting 4 of 10 right; without GW1 it gets 17 of 40 against Lasso's 18. Carrying stats over from the previous season, or using Elo, which already carries over, would fix the early gameweeks.
- The `prempredictor/` virtual environment only works on macOS (its `python` symlink points at the Mac's interpreter), and `data_fetch.py` uses an f-string with nested quotes that needs Python 3.12+.

## 2026-11-03: predict.py, and removing rejected code

`src/predict.py` predicts the next gameweek with Ordered Logit, which tied for best on the 90-match current-season test with slightly better probabilities than Lasso. It trains on every finished match on disk and writes one CSV per gameweek to `data/predictions/`. `--if-starts-tomorrow` makes it do nothing unless the next gameweek's first match is tomorrow; the daily scheduled job uses that. The next gameweek is chosen from unplayed matches dated today or later, so a postponed match can't get stuck as "next". `build_prediction_set` got an optional `week` parameter for the same reason. Predictions only go one gameweek ahead, by design.

Removed code that had been decided against (all of it is in git history at commit `4db0281`):
- XGBoost (the commented-out training code, parameter grid and `xgboost` dependency), which consistently trailed RF, Lasso and Ordered Logit.
- The RF plus XGBoost and Dixon-Coles plus Ordered Logit ensembles; neither beat its best member.
- FBref (`fbref_fetch.py`, the commented-out feature code and `data/raw/fbref/`). It never made it into the feature set because of the soccerdata bug, and the one saved file had no team on any row.
- Comments about dropped features (ppda, deep completions, the points and goal-difference pairs). The reasons are in the entries above.

Kept on purpose: Elo, odds (as an evaluation benchmark), `compute_season_weights` and Dixon-Coles (the Phase 2 starting point). Added `penaltyblog` to `requirements.txt`; `train.py` imports it, so nothing ran without it.

Deployment will use GitHub Actions and GitHub Pages rather than a scheduled job on the Mac, so predictions don't depend on the Mac being awake. That means committing `data/raw/` (~11 MB), so each run only fetches new data.

## 2026-11-05: Feature selection for Ordered Logit; xG dropped

Checked whether 13 features was too many for Ordered Logit. Feature sets were chosen using past seasons only (walk-forward: train on earlier seasons, test on each of 2023/24, 2024/25 and 2025/26, 1,140 matches, scored on log loss as well as accuracy). The candidates were then checked on 2026/27 GW1 to GW9 (90 matches, rolling), which the selection never saw.

| Feature set | # | Past: log loss / accuracy | 2026/27: log loss / correct |
|---|---|---|---|
| All | 13 | 1.0170 / 52.5% | 1.0674 / 41 of 90 |
| Without xG | 11 | 1.0184 / 51.4% | 1.0596 / 39 of 90 |
| Without venue form | 11 | 1.0108 / 52.8% | 1.0732 / 39 of 90 |
| Backward elimination (by past log loss) | 9 | 1.0147 | 1.0767 / 40 of 90 |
| Backward elimination, no xG | 7 | not recorded | 1.0685 / 38 of 90 |

Every set is within about 0.01 log loss and 2 to 4 correct matches of the others, and the set that does best on past seasons does worst on this one, and the other way round. The number of features isn't the problem, and pruning further would just fit the noise of whichever season it's tuned on.

Standardised coefficients (fitted on all past seasons) show the signal is in `home_venue_goal_diff`, `h2h_score`, `home_form`, xG and `home_rest_days` (all p < 0.025). `home_venue_form`, `away_venue_form` and `home_venue_points` are indistinguishable from zero (p > 0.45). They overlap with form and venue goal difference but don't do harm, so they stay rather than chasing a gain that depends on the season. `home_rest_days` has a negative coefficient: more rest goes with worse results. That's probably a confound, since the strongest sides play midweek in Europe and so get the least rest.

Decision: dropped `home_xg`/`away_xg`, going from 13 features to 11. Not because 13 was too many. xG's gain on past seasons is negligible (0.0014 log loss), and its only source, Understat, stopped returning data on 2026-11-04. Its pages show an error page, and soccerdata 2026.43.0 fails with a JSONDecodeError on the leagues request, with the same result after clearing the cache and upgrading. This season's xG had been stuck at GW5, and the model without it already had the better log loss on 2026/27. Dropping it also removes the least reliable step from the daily GitHub Actions job. `understat_fetch.py`, `compute_xg_form` and the saved Understat data stay, and `build_row` still calculates xG without the model using it, so it can come back if Understat recovers.

GW10 predictions were remade with the 11-feature model before kickoff. All 10 picks stayed the same; probabilities moved by up to 9 points (Man United v Villa, home win 46% down to 37%).

## Open questions

- Whether to save the built training and prediction DataFrames to `data/processed/`, or keep rebuilding them from `data/raw/` each time. Rebuilding all 5 seasons takes well under a second, so saving them may not be worth the extra complexity. Leaning towards rebuilding, not yet decided.
- Where trained models and prediction logs will live once Phase 3 (the web dashboard) starts.
- A feature for the manager, or a recent change of manager. `/lineups` does have coach data for past matches, and credits aren't a problem (50,000 available), but doing it properly needs ~1,900 calls (one per past match, since `/lineups` is keyed by match_id rather than team and season) plus real work to turn per-match coach data into something like "games since the last change of manager". Deferred until after rest days. The obstacle is engineering time, not cost.
