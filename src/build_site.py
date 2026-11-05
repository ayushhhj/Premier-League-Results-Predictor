"""Build the dashboard page from the scored results.

Reads data/results/<season>_scored.csv and <season>_summary.csv (score_results.py)
and writes site/index.html, which GitHub Actions publishes to GitHub Pages.

Usage (from src/):
    python build_site.py
"""
import html
import os
import sys
from datetime import datetime, timezone
from string import Template

import pandas as pd

from features import SEASONS

CURRENT_SEASON = SEASONS[0]


def season_tag(season):
    """"2026/2027" -> "2026-2027"."""
    return season.replace("/", "-")
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
SITE_DIR = os.path.join(BASE_DIR, "site")

# string.Template ($name placeholders) so the CSS braces don't need escaping.
# The section placeholders take ready-made HTML.
PAGE_TEMPLATE = Template("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>
  :root { --bg:#fafaf7; --fg:#1d1d1b; --muted:#6b6b66; --line:#e3e2dc; --card:#ffffff;
          --good:#1f7a4d; --bad:#b3362c; --accent:#3d3a8c; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#141413; --fg:#ecebe6; --muted:#9a998f; --line:#2c2c29; --card:#1c1c1a;
            --good:#5cc08a; --bad:#e8776c; --accent:#a9a6f2; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width:880px; margin:0 auto; padding:32px 16px 64px; }
  h1 { font-size:1.6rem; margin:0 0 4px; }
  h2 { font-size:1.1rem; margin:40px 0 12px; }
  .sub { color:var(--muted); margin:0; }
  .stat { font-size:2.2rem; font-weight:600; color:var(--accent); margin:20px 0 0; }
  .table-wrap { overflow-x:auto; background:var(--card); border:1px solid var(--line); border-radius:10px; }
  table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
  th, td { padding:8px 12px; text-align:left; border-bottom:1px solid var(--line); white-space:nowrap; }
  th { font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); font-weight:600; }
  tr:last-child td { border-bottom:none; }
  td.num, th.num { text-align:right; }
  .correct { color:var(--good); font-weight:600; }
  .pick { font-weight:700; }
  .wrong { color:var(--bad); font-weight:600; }
  .empty { color:var(--muted); padding:16px; margin:0; }
  footer { margin-top:48px; color:var(--muted); font-size:.85rem; }
  @media (max-width: 600px) {
    body { font-size:14px; }
    th, td { padding:7px 4px; }
    th { font-size:.7rem; letter-spacing:.02em; }
    .stat { font-size:1.7rem; }
    .hide-sm { display:none; }
    th, td { white-space:normal; }
  }
</style>
</head>
<body>
<main>
  <h1>$title</h1>
  <p class="sub">Premier League match predictions, one gameweek at a time. Updated $updated.</p>
  <p class="stat">$season_accuracy</p>

  <h2>Upcoming</h2>
  $upcoming

  <h2>Last gameweek</h2>
  $last_week

  <h2>Season so far</h2>
  $season_table

  <footer>Model: ordered logistic regression on form, home/away splits, head-to-head and rest days.</footer>
</main>
</body>
</html>
""")


def load_results(season=CURRENT_SEASON):
    """(scored, summary) DataFrames; empty ones if the files don't exist yet."""
    season_tag_str = season.replace("/", "-")
    scored_path = os.path.join(RESULTS_DIR, f"{season_tag_str}_scored.csv")
    summary_path = os.path.join(RESULTS_DIR, f"{season_tag_str}_summary.csv")
    if os.path.exists(scored_path):
        scored = pd.read_csv(scored_path)
    else:
        scored = pd.DataFrame(columns=["match_id", "prediction", "result", "played", "correct", "week"])
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
    else:
        summary = pd.DataFrame(columns=["week", "played", "correct", "accuracy", "home_baseline"])
    return scored, summary

def upcoming_matches(scored):
    """Unplayed matches from the earliest gameweek that has any, sorted by kickoff."""
    unplayed = scored[scored['played'] == False]
    if unplayed.empty:
        return pd.DataFrame(columns=scored.columns)
    else:
        return unplayed[unplayed['week'] == unplayed['week'].min()].sort_values(['date', 'kickoff'])

def last_played_week(scored):
    """(week, all its rows) for the latest gameweek with a played match, or (None, empty)."""
    played = scored[scored['played'] == True]
    if played.empty:
        return None, pd.DataFrame(columns=scored.columns)
    else:
        recent_week = played["week"].max()
        recent_scored = scored[scored["week"] == recent_week]
        return recent_week, recent_scored.sort_values(['date', 'kickoff'])

def rows_to_table(headers, rows, num_cols=()):
    """An HTML table from headers and rows, or a "Nothing here yet" note if rows is empty.

    A cell is plain text or a (text, css_class) tuple. Every cell is HTML-escaped.
    Columns in num_cols are right-aligned.
    """
    if not rows:
        return '<p class="empty">Nothing here yet.</p>'

    def cell_html(tag, value, index):
        # a cell is either plain text or a (text, css_class) tuple
        text, extra = value if isinstance(value, tuple) else (value, "")
        classes = " ".join(c for c in ("num" if index in num_cols else "", extra) if c)
        attr = f' class="{classes}"' if classes else ""
        return f"<{tag}{attr}>{html.escape(str(text))}</{tag}>"

    out = ['<div class="table-wrap"><table>', "<thead><tr>"]
    out += [cell_html("th", h, i) for i, h in enumerate(headers)]
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>" + "".join(cell_html("td", c, i) for i, c in enumerate(row)) + "</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)

def outcome_name(code, home, away):
    """H/D/A -> what a reader expects to see: the winning team's name, or "Draw"."""
    return {"H": home, "D": "Draw", "A": away}.get(code, "")


def when(date_str, kickoff):
    """("2026-11-07", "15:00") -> "Sat 7 Nov, 15:00"."""
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return f"{d.strftime('%a')} {d.day} {d.strftime('%b')}, {kickoff}"


def pct(x):
    """0.483 -> "48%"; blank for missing values."""
    return "–" if pd.isna(x) else f"{x * 100:.0f}%"


def build_upcoming_html(upcoming):
    """Upcoming gameweek table. On phones the kickoff and pick columns are hidden and the picked probability is bold."""
    if upcoming.empty:
        return '<p class="empty">No upcoming matches.</p>'
    week = int(upcoming["week"].iloc[0])
    rows = []
    for _, m in upcoming.iterrows():
        # bold the picked probability; it stands in for the Pick column on phones
        probs = [(pct(m[col]), "pick" if m["prediction"] == code else "")
                 for col, code in (("p_home", "H"), ("p_draw", "D"), ("p_away", "A"))]
        rows.append([
            (when(m["date"], m["kickoff"]), "hide-sm"), m["home_team"], m["away_team"],
            *probs,
            (outcome_name(m["prediction"], m["home_team"], m["away_team"]), "hide-sm"),
        ])
    table = rows_to_table(
        [("Kickoff", "hide-sm"), "Home", "Away", "Home win", "Draw", "Away win", ("Pick", "hide-sm")],
        rows, num_cols=(3, 4, 5),
    )
    return f'<p class="sub">Gameweek {week}</p>\n{table}'

def build_last_week_html(week, rows):
    """Last gameweek's picks against results, with a tick or cross per played match."""
    if week is None or rows.empty:
        return '<p class="empty">No gameweek played yet.</p>'
    table_rows = []
    for _, m in rows.iterrows():
        home, away = m["home_team"], m["away_team"]
        if m["played"]:
            score = f"{int(m['home_score'])}–{int(m['away_score'])}"
            result = outcome_name(m["result"], home, away)
            mark = ("✓", "correct") if m["correct"] else ("✗", "wrong")
        else:
            score, result, mark = "–", "Not played yet", ""
        table_rows.append([home, away, outcome_name(m["prediction"], home, away), score, (result, "hide-sm"), mark])
    table = rows_to_table(["Home", "Away", "Pick", "Score", ("Result", "hide-sm"), ""], table_rows, num_cols=(3,))
    return f'<p class="sub">Gameweek {int(week)}</p>\n{table}'

def build_season_html(summary):
    """Played, correct, accuracy and always-home baseline per gameweek and for the season."""
    if summary.empty or summary["played"].sum() == 0:
        return '<p class="empty">No results yet.</p>'
    rows = []
    for _, r in summary.iterrows():
        label = "Season" if str(r["week"]) == "Season" else f"GW{r['week']}"
        if int(r["played"]) == 0:
            continue
        acc = "–" if pd.isna(r["accuracy"]) else f"{r['accuracy'] * 100:.1f}%"
        base = "–" if pd.isna(r.get("home_baseline")) else f"{r['home_baseline'] * 100:.1f}%"
        rows.append([label, int(r["played"]), int(r["correct"]), acc, base])
    return rows_to_table(
        ["Week", "Played", "Correct", "Accuracy", "Always home"],
        rows, num_cols=(1, 2, 3, 4),
    )

def season_accuracy_text(summary):
    """Headline text, e.g. "41 / 90 correct (45.6%)", or "No results yet"."""
    season = summary[summary["week"].astype(str) == "Season"] if not summary.empty else summary
    if season.empty or int(season["played"].iloc[0]) == 0:
        return "No results yet"
    played, correct = int(season["played"].iloc[0]), int(season["correct"].iloc[0])
    return f"{correct} / {played} correct ({correct / played * 100:.1f}%)"

def write_site(page_html):
    """Write the page to SITE_DIR/index.html and return the path."""
    os.makedirs(SITE_DIR, exist_ok=True)
    path = os.path.join(SITE_DIR, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page_html)
    return path


def main():
    """Command-line entry point. Returns the exit code."""

    scored, summary = load_results()
    upcoming = upcoming_matches(scored)
    last_week_num, last_week_rows = last_played_week(scored)
    page_html = PAGE_TEMPLATE.substitute(
        title=f"Prem Predictor {CURRENT_SEASON}",
        updated=datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC"),
        season_accuracy=season_accuracy_text(summary),
        upcoming=build_upcoming_html(upcoming),
        last_week=build_last_week_html(last_week_num, last_week_rows),
        season_table=build_season_html(summary)
    )
    path = write_site(page_html)
    print(f"Site written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
