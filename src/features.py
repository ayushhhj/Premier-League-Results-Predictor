import datetime
import json
import os
import pandas
from data_fetch import _season_tag

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
MATCHES_DIR = os.path.join(DATA_DIR, "matches")
TEAM_MATCHES_DIR = os.path.join(DATA_DIR, "team_matches")
SEASONS = ["2026/2027", "2025/2026", "2024/2025", "2023/2024", "2022/2023"]
EXCLUDED_COMPETITIONS = {"Club Friendly"}  # preseason friendlies don't reflect real fatigue

# --- Loading raw data saved by the fetch scripts ---

def load_fixtures(season):
    """Load a season's saved league_fixtures JSON (one API response per gameweek)."""
    fixtures_path = os.path.join(DATA_DIR, f"league_fixtures_{_season_tag(season)}.json")
    with open(fixtures_path, "r") as f:
        return json.load(f)


def load_standings(season):
    """Load a season's saved league_standings JSON."""
    standings_path = os.path.join(DATA_DIR, f"league_standings_{_season_tag(season)}.json")
    with open(standings_path, "r") as f:
        return json.load(f)


def load_h2h(match_id):
    """Load the saved head-to-head JSON for a match."""
    h2h_path = os.path.join(MATCHES_DIR, f"h2h_{match_id}.json")
    with open(h2h_path, "r") as f:
        return json.load(f)


def load_team_matches(team_id, season):
    """A team's matches across all competitions for a season (used for rest days)."""
    path = os.path.join(TEAM_MATCHES_DIR, f"{team_id}_{_season_tag(season)}.json")
    with open(path) as f:
        data = json.load(f)
    return data["data"]["matches"]


def build_team_matches_lookup(season, season_matches):
    """{team_id: matches} for every team in season_matches; an empty list if a team's file is missing."""
    team_ids = {m["home"]["id"] for m in season_matches} | {m["away"]["id"] for m in season_matches}
    lookup = {}
    for team_id in team_ids:
        try:
            lookup[team_id] = load_team_matches(team_id, season)
        except FileNotFoundError:
            lookup[team_id] = []
    return lookup


def compute_rest_days(before_date, team_matches):
    """Days since the team's last competitive match before before_date.

    Counts every competition except preseason friendlies. Returns 7 if there's no
    earlier match.
    """
    prior_dates = [
        match["date"][:10]  # "date" here is "YYYY-MM-DD HH:MM", unlike fixtures' plain date
        for match in team_matches
        if match["date"][:10] < before_date and match["league"]["name"] not in EXCLUDED_COMPETITIONS
    ]
    if not prior_dates:
        return 7

    most_recent = max(prior_dates)
    days = (datetime.date.fromisoformat(before_date) - datetime.date.fromisoformat(most_recent)).days
    return days


UNDERSTAT_DIR = os.path.join(DATA_DIR, "understat")

# Understat team name -> our team_id (names differ between the two sources).
UNDERSTAT_TEAM_MAP = {
    "Arsenal": "4dsgumo7d4zupm2ugsvm4zm4d",
    "Aston Villa": "b496gs285it6bheuikox6z9mj",
    "Bournemouth": "1pse9ta7a45pi2w2grjim70ge",
    "Brentford": "7yx5dqhhphyvfisohikodajhv",
    "Brighton": "e5p0ehyguld7egzhiedpdnc3w",
    "Burnley": "64bxxwu2mv2qqlv0monbkj1om",
    "Chelsea": "9q0arba2kbnywth8bkxlhgmdr",
    "Coventry": "dacoc20k0n3fxu4zek8wufmko",
    "Crystal Palace": "1c8m2ko0wxq1asfkuykurdr0y",
    "Everton": "ehd2iemqmschhj2ec0vayztzz",
    "Fulham": "hzqh7z0mdl3v7gwete66syxp",
    "Hull": "f0figafm2v9h8ofo6slajj7xf",
    "Ipswich": "8b523ujgl21tbc01me65q0aoh",
    "Leeds": "48gk2hpqtsl6p9sx9kjhaydq4",
    "Leicester": "avxknfz4f6ob0rv9dbnxdzde0",
    "Liverpool": "c8h9bw1l82s06h77xxrelzhur",
    "Luton": "aksa492u5hf93giwcn2zt1nzz",
    "Manchester City": "a3nyxabgsqlnqfkeg41m6tnpp",
    "Manchester United": "6eqit8ye8aomdsrrq0hk3v7gh",
    "Newcastle United": "7vn2i2kd35zuetw6b38gw9jsz",
    "Nottingham Forest": "1qtaiy11gswx327s0vkibf70n",
    "Sheffield United": "bws31egwjda253q9lvykgnivo",
    "Southampton": "d5ydtvt96bv7fq04yqm2w2632",
    "Sunderland": "1r3545b2dzan8yqa80gtmcjch",
    "Tottenham": "22doj4sgsocqpxw45h607udje",
    "West Ham": "4txjdaqveermfryvbfrr4taf7",
    "Wolverhampton Wanderers": "b9si1jn1lfxfund69e9ogcu2n",
}


def list_own_team_names(seasons):
    """{team_id: name} for every team in the given seasons' fixtures. Used to build the team-name maps."""
    names = {}
    for season in seasons:
        for match in flatten_fixtures(season):
            names[match["home"]["id"]] = match["home"]["name"]
            names[match["away"]["id"]] = match["away"]["name"]
    return names


def list_understat_team_names(seasons):
    """Every team name in the saved Understat data for the given seasons."""
    names = set()
    for season in seasons:
        for match in load_understat_matches(season):
            names.add(match["home_team"])
            names.add(match["away_team"])
    return names


def load_understat_matches(season):
    """Load a season's saved Understat match stats (understat_fetch.py)."""
    path = os.path.join(UNDERSTAT_DIR, f"{_season_tag(season)}.json")
    with open(path) as f:
        return json.load(f)


def understat_team_matches_before(team_id, before_date, understat_season_matches):
    """The team's Understat matches dated before before_date."""
    reverse_map = {team_id_: name for name, team_id_ in UNDERSTAT_TEAM_MAP.items()}
    team_name = reverse_map.get(team_id)
    if team_name is None:
        return []

    matches = []
    for match in understat_season_matches:
        if match["date"][:10] >= before_date:
            continue
        if match["home_team"] == team_name or match["away_team"] == team_name:
            matches.append(match)
    return matches


def compute_xg_form(team_id, before_date, understat_season_matches, metric="xg", n=5):
    """Average of `metric` (e.g. "xg") over the team's last n Understat matches before before_date; 0.0 if none."""
    reverse_map = {team_id_: name for name, team_id_ in UNDERSTAT_TEAM_MAP.items()}
    team_name = reverse_map.get(team_id)
    if team_name is None:
        return 0.0

    matches = understat_team_matches_before(team_id, before_date, understat_season_matches)[::-1][:n]
    if not matches:
        return 0.0

    values = []
    for match in matches:
        if match["home_team"] == team_name:
            values.append(match[f"home_{metric}"])
        else:
            values.append(match[f"away_{metric}"])

    return sum(values) / len(values)


ODDS_DIR = os.path.join(DATA_DIR, "odds")

# Football-Data.co.uk team name -> our team_id.
ODDS_TEAM_MAP = {
    "Arsenal": "4dsgumo7d4zupm2ugsvm4zm4d",
    "Aston Villa": "b496gs285it6bheuikox6z9mj",
    "Bournemouth": "1pse9ta7a45pi2w2grjim70ge",
    "Brentford": "7yx5dqhhphyvfisohikodajhv",
    "Brighton": "e5p0ehyguld7egzhiedpdnc3w",
    "Burnley": "64bxxwu2mv2qqlv0monbkj1om",
    "Chelsea": "9q0arba2kbnywth8bkxlhgmdr",
    "Coventry": "dacoc20k0n3fxu4zek8wufmko",
    "Crystal Palace": "1c8m2ko0wxq1asfkuykurdr0y",
    "Everton": "ehd2iemqmschhj2ec0vayztzz",
    "Fulham": "hzqh7z0mdl3v7gwete66syxp",
    "Hull": "f0figafm2v9h8ofo6slajj7xf",
    "Ipswich": "8b523ujgl21tbc01me65q0aoh",
    "Leeds": "48gk2hpqtsl6p9sx9kjhaydq4",
    "Leicester": "avxknfz4f6ob0rv9dbnxdzde0",
    "Liverpool": "c8h9bw1l82s06h77xxrelzhur",
    "Luton": "aksa492u5hf93giwcn2zt1nzz",
    "Man City": "a3nyxabgsqlnqfkeg41m6tnpp",
    "Man United": "6eqit8ye8aomdsrrq0hk3v7gh",
    "Newcastle": "7vn2i2kd35zuetw6b38gw9jsz",
    "Nott'm Forest": "1qtaiy11gswx327s0vkibf70n",
    "Sheffield United": "bws31egwjda253q9lvykgnivo",
    "Southampton": "d5ydtvt96bv7fq04yqm2w2632",
    "Sunderland": "1r3545b2dzan8yqa80gtmcjch",
    "Tottenham": "22doj4sgsocqpxw45h607udje",
    "West Ham": "4txjdaqveermfryvbfrr4taf7",
    "Wolves": "b9si1jn1lfxfund69e9ogcu2n",
}


def load_odds(season):
    """Load a season's saved odds (odds_fetch.py): date, teams and average H/D/A decimal odds."""
    path = os.path.join(ODDS_DIR, f"{_season_tag(season)}.json")
    with open(path) as f:
        return json.load(f)


def build_odds_lookup(season):
    """{(home_id, away_id, date): (p_home, p_draw, p_away)} for a season.

    Probabilities are 1/odds normalised to sum to 1, which removes the bookmaker's margin.
    """
    lookup = {}
    for row in load_odds(season):
        home_id = ODDS_TEAM_MAP.get(row["home_team"])
        away_id = ODDS_TEAM_MAP.get(row["away_team"])
        if home_id is None or away_id is None:
            continue

        inv_h, inv_d, inv_a = 1 / row["AvgH"], 1 / row["AvgD"], 1 / row["AvgA"]
        total = inv_h + inv_d + inv_a

        date = row["date"][:10]
        lookup[(home_id, away_id, date)] = (inv_h / total, inv_d / total, inv_a / total)

    return lookup


def get_match_odds(home_id, away_id, date, odds_lookup):
    """(p_home, p_draw, p_away) for a match; an even 1/3 split if there are no odds for it."""
    return odds_lookup.get((home_id, away_id, date), (1 / 3, 1 / 3, 1 / 3))


ELO_INITIAL_RATING = 1500
ELO_K_FACTOR = 20  # untuned
ELO_HOME_ADVANTAGE = 100  # added to the home rating when computing the expected score


def expected_score(rating_a, rating_b):
    """Elo expected score (win probability) for a side rated rating_a against one rated rating_b."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def elo_result_value(home_score, away_score):
    """Elo actual score from the home side's view: 1 for a win, 0.5 for a draw, 0 for a loss."""
    elo_score = 0.0
    if home_score > away_score:
        elo_score += 1.0
    elif home_score == away_score:
        elo_score += 0.5
    
    return elo_score



def all_matches_chronological(seasons):
    """Every finished match across the given seasons, oldest first.

    Elo runs over this in one pass because ratings carry over between seasons.
    """
    matches = []
    fixtures = []
    for season in seasons:
        fixtures.extend(flatten_fixtures(season))
    
    for match in fixtures:
        if match["status"]["status"] == "finished":
            matches.append(match)

    return sorted(matches, key=lambda match: match['date'])


def compute_elo_lookup(seasons, k=ELO_K_FACTOR, home_advantage=ELO_HOME_ADVANTAGE, initial_rating=ELO_INITIAL_RATING):
    """Run Elo over every finished match in date order.

    Returns (lookup, ratings):
    - lookup: match_id -> (home_elo, away_elo) going into that match, for finished matches
    - ratings: team_id -> latest rating, for unplayed matches

    Ratings carry over between seasons unchanged.
    """
    ratings = {}
    lookup = {}
    for match in all_matches_chronological(seasons):
        home_id = match["home"]["id"]
        away_id = match["away"]["id"]
        home_score = match["home"]["score"]
        away_score = match["away"]["score"]

        home_rating = ratings.get(home_id, initial_rating)
        away_rating = ratings.get(away_id, initial_rating)

        lookup[match["id"]] = (home_rating, away_rating)

        expected_home = expected_score(home_rating + home_advantage, away_rating)
        actual_home = elo_result_value(home_score, away_score)

        new_home_rating = home_rating + k * (actual_home - expected_home)
        new_away_rating = away_rating + k * ((1 - actual_home) - (1 - expected_home))

        ratings[home_id] = new_home_rating
        ratings[away_id] = new_away_rating

    return lookup, ratings



def get_match_elo(match_id, elo_lookup, initial_rating=ELO_INITIAL_RATING):
    """(home_elo, away_elo) going into a finished match; the initial rating if it isn't in the lookup."""
    return elo_lookup.get(match_id, (initial_rating, initial_rating))


def flatten_fixtures(season):
    """A season's fixtures as a flat list of match dicts, each tagged with an integer `week`.

    Skips gameweeks whose fetch failed.
    """
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


# --- Point-in-time features: only matches before the given date are used ---

def team_matches_before(team_id, before_date, season_matches, venue=None):
    """The team's matches in season_matches dated before before_date.

    venue: None for all matches, or "home"/"away" for only that venue.
    """
    matches = []
    for match in season_matches:
        if match["date"] >= before_date:
            continue
        is_home = match["home"]["id"] == team_id
        is_away = match["away"]["id"] == team_id
        if not (is_home or is_away):
            continue
        if venue == "home" and not is_home:
            continue
        if venue == "away" and not is_away:
            continue
        matches.append(match)
    return matches


def compute_form(team_id, before_date, season_matches, venue=None, decay=0.8):
    """Recency-weighted form from the team's last 5 matches before before_date, from 0 to 1.

    The i-th most recent match is weighted decay**i, normalised by the weights actually
    used so the scale holds with fewer than 5 matches. venue: as in team_matches_before.
    """
    matches = team_matches_before(team_id, before_date, season_matches, venue=venue)

    weighted_points = 0.0
    weighted_max = 0.0
    for i, match in enumerate(matches[::-1][:5]):
        weight = decay ** i
        if match["home"]["id"] == team_id:
            team_score, opponent_score = int(match["home"]["score"]), int(match["away"]["score"])
        else:
            team_score, opponent_score = int(match["away"]["score"]), int(match["home"]["score"])

        if team_score > opponent_score:
            weighted_points += 3 * weight
        elif team_score == opponent_score:
            weighted_points += 1 * weight

        weighted_max += 3 * weight

    if weighted_max == 0:
        return 0.0
    return weighted_points / weighted_max

def compute_goal_stats(team_id, before_date, season_matches, venue=None):
    """(points, goal_difference) from the team's matches this season before before_date. venue: as in team_matches_before."""
    goals_scored = 0
    goals_conceded = 0
    points = 0
    matches = team_matches_before(team_id, before_date, season_matches, venue=venue)

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
    """Head-to-head meetings dated before before_date."""
    return [match for match in h2h_list if match["date"] < before_date]


def build_h2h_lookup(current_season=SEASONS[0]):
    """{frozenset({team_a, team_b}): h2h data} from the saved h2h files.

    h2h is only fetched for the current season's fixtures. Each file holds the pairing's
    full history, so it serves every season once filtered by date.
    """
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
    """(home_wins, away_wins, draws, h2h_score) for meetings before before_date.

    h2h_score is (home_wins - away_wins) / meetings. Counted from the match list rather
    than the API's h2h_summary, which is oriented to whichever team was home in the
    fixture the data was fetched for.
    """
    h2h_data = h2h_lookup.get(frozenset({home_id, away_id}))
    if h2h_data is None:
        return 0, 0, 0, 0.0

    h2h_list = filter_h2h_before_date(h2h_data["data"]["h2h"], before_date)

    home_wins = away_wins = draws = 0
    h2h_score = 0.0
    
    for match in h2h_list:
        # h2h scores are a single "H-A" string
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

    if len(h2h_list) != 0:    
        h2h_score = (home_wins - away_wins) / len(h2h_list)

    return home_wins, away_wins, draws, h2h_score


# --- dataset construction ---

def build_row(match, season_matches, h2h_lookup, season, team_matches_lookup, understat_season_matches, odds_lookup, elo_lookup, current_elo_ratings):
    """One feature row for a match, plus the score and result label if it's finished."""
    home_id = match["home"]["id"]
    away_id = match["away"]["id"]
    date = match["date"]

    home_form = compute_form(home_id, date, season_matches)
    away_form = compute_form(away_id, date, season_matches)

    home_venue_form = compute_form(home_id, date, season_matches, venue="home")
    away_venue_form = compute_form(away_id, date, season_matches, venue="away")
    home_venue_points, home_venue_gd = compute_goal_stats(home_id, date, season_matches, venue="home")
    away_venue_points, away_venue_gd = compute_goal_stats(away_id, date, season_matches, venue="away")

    h2h_home_wins, h2h_away_wins, h2h_draws, h2h_score = compute_h2h_stats(home_id, away_id, date, h2h_lookup)

    home_rest_days = compute_rest_days(date, team_matches_lookup.get(home_id, []))
    away_rest_days = compute_rest_days(date, team_matches_lookup.get(away_id, []))

    home_xg = compute_xg_form(home_id, date, understat_season_matches, metric="xg")
    away_xg = compute_xg_form(away_id, date, understat_season_matches, metric="xg")

    market_prob_home, market_prob_draw, market_prob_away = get_match_odds(home_id, away_id, date, odds_lookup)

    if match["status"]["status"] == "finished":
        # rating going into this match
        home_elo, away_elo = get_match_elo(match["id"], elo_lookup)
    else:
        # not played yet: latest known rating
        home_elo = current_elo_ratings.get(home_id, ELO_INITIAL_RATING)
        away_elo = current_elo_ratings.get(away_id, ELO_INITIAL_RATING)

    row = {
        "match_id": match["id"],
        "season": season,
        "date": date,
        "home_team": home_id,
        "away_team": away_id,
        "home_form": home_form,
        "away_form": away_form,
        "home_venue_form": home_venue_form,
        "away_venue_form": away_venue_form,
        "home_venue_points": home_venue_points,
        "away_venue_points": away_venue_points,
        "home_venue_goal_diff": home_venue_gd,
        "away_venue_goal_diff": away_venue_gd,
        "h2h_score": h2h_score,
        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
        "home_xg": home_xg,
        "away_xg": away_xg,
        "market_prob_home": market_prob_home,
        "market_prob_draw": market_prob_draw,
        "market_prob_away": market_prob_away,
        "home_elo": home_elo,
        "away_elo": away_elo,
    }

    if match["status"]["status"] == "finished":
        home_score = int(match["home"]["score"])
        away_score = int(match["away"]["score"])
        row["home_score"] = home_score
        row["away_score"] = away_score
        if home_score > away_score:
            row["result"] = "H"
        elif away_score > home_score:
            row["result"] = "A"
        else:
            row["result"] = "D"

    return row


def build_training_set(seasons):
    """Feature rows with result labels for every finished match in the given seasons."""
    h2h_lookup = build_h2h_lookup()
    # 2021/22 only warms up Elo ratings; it never becomes training rows
    elo_lookup, current_elo_ratings = compute_elo_lookup(["2021/2022"] + seasons)

    rows = []
    for season in seasons:
        season_matches = flatten_fixtures(season)
        team_matches_lookup = build_team_matches_lookup(season, season_matches)
        try:
            understat_season_matches = load_understat_matches(season)
        except FileNotFoundError:
            understat_season_matches = []
        try:
            odds_lookup = build_odds_lookup(season)
        except FileNotFoundError:
            odds_lookup = {}
        for match in season_matches:
            if match["status"]["status"] != "finished":
                continue
            rows.append(build_row(
                match, season_matches, h2h_lookup, season, team_matches_lookup,
                understat_season_matches, odds_lookup, elo_lookup, current_elo_ratings
            ))

    return pandas.DataFrame(rows)


def build_prediction_set(season, week=None):
    """Feature rows for the next unplayed gameweek, or for `week` if given.

    Only one gameweek ahead: later gameweeks' features depend on results not played yet.
    """
    h2h_lookup = build_h2h_lookup(season)
    season_matches = flatten_fixtures(season)
    team_matches_lookup = build_team_matches_lookup(season, season_matches)
    try:
        understat_season_matches = load_understat_matches(season)
    except FileNotFoundError:
        understat_season_matches = []
    try:
        odds_lookup = build_odds_lookup(season)
    except FileNotFoundError:
        odds_lookup = {}
    # Elo ratings carry across seasons, so use the full history
    elo_lookup, current_elo_ratings = compute_elo_lookup(["2021/2022"] + SEASONS)

    unplayed = [m for m in season_matches if m["status"]["status"] != "finished"]
    if not unplayed:
        return pandas.DataFrame([])

    if week is not None:
        next_week = week
    else:
        next_week = min(m["week"] for m in unplayed)

    rows = [
        build_row(
            match, season_matches, h2h_lookup, season, team_matches_lookup,
            understat_season_matches, odds_lookup, elo_lookup, current_elo_ratings
        )
        for match in unplayed
        if match["week"] == next_week
    ]
    return pandas.DataFrame(rows)


if __name__ == "__main__":
    pass
