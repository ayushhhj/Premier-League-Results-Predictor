import json
import os
import pandas
from data_fetch import _season_tag

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
MATCHES_DIR = os.path.join(DATA_DIR, "matches")
SEASONS = ["2026/2027", "2025/2026", "2024/2025", "2023/2024", "2022/2023"]

# --- Loading raw data already saved by data_fetch.py ---

def load_fixtures(season):
    """Load the saved league_fixtures JSON for a season, return the flat list of match dicts."""
    fixtures_path = os.path.join(DATA_DIR, f"league_fixtures_{_season_tag(season)}.json")
    with open(fixtures_path, "r") as f:
        return json.load(f)


def load_standings(season):
    """Load the saved league_standings JSON for a season."""
    standings_path = os.path.join(DATA_DIR, f"league_standings_{_season_tag(season)}.json")
    with open(standings_path, "r") as f:
        return json.load(f)


def load_h2h(match_id):
    """Load the saved h2h JSON for a given match_id (data/raw/matches/h2h_<id>.json)."""
    h2h_path = os.path.join(MATCHES_DIR, f"h2h_{match_id}.json")
    with open(h2h_path, "r") as f:
        return json.load(f)


def flatten_fixtures(season):
    """Load a season's fixtures and flatten the nested weeks->matches structure into
    a single flat list of match dicts (what all the point-in-time functions expect).
    Skips any week that failed to fetch (missing "data", e.g. a rate-limited call)."""
    raw = load_fixtures(season)
    matches = []
    for week_response in raw:
        if "data" not in week_response:
            continue
        for week in week_response["data"]["weeks"]:
            for match in week["matches"]:
                match["week"] = int(week["week"])  # API gives this as a string
                matches.append(match)
    return matches


# --- Point-in-time computation ---
# Reminder: standings/form for a training row must reflect what was true
# BEFORE that match was played, not the final/current-day standings.

def team_matches_before(team_id, before_date, season_matches):
    """Return a team's matches from season_matches that happened strictly before before_date.

    season_matches is expected to be a FLAT list of match dicts (each with "date", "home",
    "away", etc. — see the "matches" list inside load_fixtures()'s week entries), not the
    raw nested load_fixtures() output. Dates are ISO "YYYY-MM-DD" strings, so plain string
    comparison sorts them correctly without needing to parse them into date objects.
    """
    matches = []
    for match in season_matches:
        if match["date"] >= before_date:
            continue
        if match["home"]["id"] == team_id or match["away"]["id"] == team_id:
            matches.append(match)
    return matches


def compute_form(team_id, before_date, season_matches):
    """Compute a team's form (e.g. points from last 5 matches) as of before_date."""
    points = 0
    i=0
    matches = team_matches_before(team_id, before_date, season_matches)

    for match in matches[::-1]:
        if match["home"]["id"] == team_id:
            if int(match["home"]["score"]) > int(match["away"]["score"]):
                points+=3
            elif int(match["home"]["score"]) == int(match["away"]["score"]):
                points+=1
            i+=1
        elif match["away"]["id"] == team_id:
            if int(match["away"]["score"]) > int(match["home"]["score"]):
                points+=3
            elif int(match["away"]["score"]) == int(match["home"]["score"]):
                points+=1
            i+=1
        if i==5:
            break

    form = points / 15
    return form

def compute_goal_stats(team_id, before_date, season_matches):
    """Compute a team's total points, goals scored/conceded as of before_date, derived from results
    so far that season (same point-in-time logic as compute_form)."""
    goals_scored = 0
    goals_conceded = 0
    points = 0
    matches = team_matches_before(team_id, before_date, season_matches)

    for match in matches:
        if match["home"]["id"] == team_id:
           team_score = int(match["home"]["score"])
           opponent_score = int(match["away"]["score"])
        else:
            team_score = int(match["away"]["score"])
            opponent_score = int(match["home"]["score"])
        
        goals_scored += team_score
        goals_conceded += opponent_score

        if team_score > opponent_score:
            points += 3
        elif team_score == opponent_score:
            points +=1

    goal_difference = goals_scored - goals_conceded
    return points, goal_difference

def filter_h2h_before_date(h2h_list, before_date):
    """Filter an h2h match list down to only meetings that happened before before_date."""
    return [match for match in h2h_list if match["date"] < before_date]


def build_h2h_lookup(current_season=SEASONS[0]):
    """Build a {frozenset({team_a_id, team_b_id}): h2h_data} lookup from saved h2h files.
    h2h was only ever pulled for the current season's matches (per the h2h-scope decision
    in docs/DECISIONS.md), so this is the single source used for every historical match
    between the same pairing, regardless of which season that match is actually in."""
    matches = flatten_fixtures(current_season)
    lookup = {}
    for match in matches:
        key = frozenset({match["home"]["id"], match["away"]["id"]})
        if key in lookup:
            continue  # already have this pairing's h2h data from the other fixture
        try:
            lookup[key] = load_h2h(match["id"])
        except FileNotFoundError:
            continue
    return lookup


def compute_h2h_stats(home_id, away_id, before_date, h2h_lookup):
    """Return (home_wins, away_wins, draws) between home_id/away_id as of before_date.

    Recomputed fresh from the raw h2h match list rather than trusting the API's own
    h2h_summary field, since that summary is oriented to whichever team was "home" when
    the h2h data was originally fetched — which may be the opposite of home_id/away_id
    here if this pairing's other (reverse-fixture) match_id was used to fetch it.
    """
    h2h_data = h2h_lookup.get(frozenset({home_id, away_id}))
    if h2h_data is None:
        return 0, 0, 0

    h2h_list = filter_h2h_before_date(h2h_data["data"]["h2h"], before_date)

    home_wins = away_wins = draws = 0
    for match in h2h_list:
        # h2h entries use a combined "H-A" score string on the match itself,
        # unlike fixtures data where each side has its own "score" field.
        match_home_score, match_away_score = (int(s) for s in match["score"].split("-"))
        if match["home"]["id"] == home_id:
            home_id_score, away_id_score = match_home_score, match_away_score
        else:
            home_id_score, away_id_score = match_away_score, match_home_score

        if home_id_score > away_id_score:
            home_wins += 1
        elif away_id_score > home_id_score:
            away_wins += 1
        else:
            draws += 1

    return home_wins, away_wins, draws


# --- dataset construction ---

def build_row(match, season_matches, h2h_lookup):
    """Build one feature row (+ label, if the match is finished) for a single match."""
    home_id = match["home"]["id"]
    away_id = match["away"]["id"]
    date = match["date"]

    home_form = compute_form(home_id, date, season_matches)
    away_form = compute_form(away_id, date, season_matches)
    home_points, home_gd = compute_goal_stats(home_id, date, season_matches)
    away_points, away_gd = compute_goal_stats(away_id, date, season_matches)
    h2h_home_wins, h2h_away_wins, h2h_draws = compute_h2h_stats(home_id, away_id, date, h2h_lookup)

    row = {
        "match_id": match["id"],
        "date": date,
        "home_team": home_id,
        "away_team": away_id,
        "home_form": home_form,
        "away_form": away_form,
        "home_points": home_points,
        "away_points": away_points,
        "home_goal_diff": home_gd,
        "away_goal_diff": away_gd,
        "h2h_home_wins": h2h_home_wins,
        "h2h_away_wins": h2h_away_wins,
        "h2h_draws": h2h_draws,
    }

    if match["status"]["status"] == "finished":
        home_score = int(match["home"]["score"])
        away_score = int(match["away"]["score"])
        if home_score > away_score:
            row["result"] = "H"
        elif away_score > home_score:
            row["result"] = "A"
        else:
            row["result"] = "D"

    return row


def build_training_set(seasons):
    """Loop over all *finished* matches across the given seasons and build the full
    training DataFrame (features + actual outcome label)."""
    h2h_lookup = build_h2h_lookup()  # current season, per the h2h-scope decision

    rows = []
    for season in seasons:
        season_matches = flatten_fixtures(season)
        for match in season_matches:
            if match["status"]["status"] != "finished":
                continue
            rows.append(build_row(match, season_matches, h2h_lookup))

    return pandas.DataFrame(rows)


def build_prediction_set(season):
    """Build feature rows (no label) for the NEXT unplayed matchweek only.

    Can't go further than that: computing point-in-time features for a fixture several
    weeks out would require already knowing the results of the weeks in between, which
    haven't been played yet either. This is also what makes the weekly-refresh dashboard
    idea work — predictions get regenerated one matchweek at a time as results come in.
    """
    h2h_lookup = build_h2h_lookup(season)
    season_matches = flatten_fixtures(season)

    unplayed = [m for m in season_matches if m["status"]["status"] != "finished"]
    if not unplayed:
        return pandas.DataFrame([])
    next_week = min(m["week"] for m in unplayed)

    rows = [
        build_row(match, season_matches, h2h_lookup)
        for match in unplayed
        if match["week"] == next_week
    ]
    return pandas.DataFrame(rows)


if __name__ == "__main__":
    pass
