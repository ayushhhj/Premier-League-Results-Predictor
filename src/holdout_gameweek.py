"""Test the model on the current season, one gameweek at a time.

Holds out a gameweek, trains on everything played before it (all past seasons plus
this season's earlier gameweeks) and scores each model on the held-out matches, with
bookmaker odds as a reference.

- default: the latest finished gameweek (or --week N)
- --rolling: every finished gameweek in turn, pooled. One gameweek is only 10
  matches, so the pooled result is the one to go by.

Usage (from src/):
    python holdout_gameweek.py
    python holdout_gameweek.py --week 4
    python holdout_gameweek.py --rolling
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from features import build_training_set, flatten_fixtures, SEASONS
from train import (
    LABEL_MAP, FEATURE_COLUMNS, get_features_and_labels,
    train_baseline, train_random_forest, train_lasso, train_ordered_logit,
    train_dixon_coles, get_ordinal_features_and_labels,
)

warnings.filterwarnings("ignore")
CURRENT_SEASON = SEASONS[0]
OUTCOMES = ["H", "D", "A"]  # column order for every probability matrix below


def load_all_rows():
    """Every finished match across SEASONS with its gameweek and team names."""
    df = build_training_set(SEASONS)
    week_of, name_of = {}, {}
    for season in SEASONS:
        for m in flatten_fixtures(season):
            week_of[m["id"]] = m["week"]
            name_of[m["home"]["id"]] = m["home"]["name"]
            name_of[m["away"]["id"]] = m["away"]["name"]
    df["week"] = df["match_id"].map(week_of)
    df["home_name"] = df["home_team"].map(name_of)
    df["away_name"] = df["away_team"].map(name_of)
    return df


def split(df, week):
    is_current = df["season"] == CURRENT_SEASON
    train_df = df[~is_current | (df["week"] < week)]
    test_df = df[is_current & (df["week"] == week)]
    return train_df, test_df


def predict_all(train_df, test_df):
    """Fit every model on train_df. Returns {model name: (n x 3) probabilities in H/D/A order}."""
    X_test, _ = get_features_and_labels(test_df)
    probs = {}

    baseline = train_baseline(train_df)
    p = baseline.predict_proba(X_test)
    probs["Baseline (always home)"] = np.eye(3)[np.argmax(p, axis=1)] * 0.999 + 0.0005

    rf = train_random_forest(train_df)
    probs["Random Forest"] = rf.predict_proba(X_test)  # classes 0/1/2 = H/D/A

    lasso, scaler = train_lasso(train_df)
    probs["Lasso"] = lasso.predict_proba(scaler.transform(X_test))

    ol = train_ordered_logit(train_df)
    X_ord, _ = get_ordinal_features_and_labels(test_df)
    probs["Ordered Logit"] = ol.model.predict(ol.params, exog=X_ord)[:, [2, 1, 0]]  # A/D/H -> H/D/A

    dc = train_dixon_coles(train_df)
    known = set(train_df["home_team"]) | set(train_df["away_team"])
    dc_rows = []
    for _, r in test_df.iterrows():
        if r["home_team"] in known and r["away_team"] in known:
            dc_rows.append(list(dc.predict(r["home_team"], r["away_team"]).home_draw_away))
        else:
            dc_rows.append([1 / 3] * 3)
    probs["Dixon-Coles"] = np.array(dc_rows)

    # reference only, not a model input
    probs["Bookmakers (reference)"] = test_df[
        ["market_prob_home", "market_prob_draw", "market_prob_away"]
    ].to_numpy()
    return probs


def rps(probs, y):
    """Ranked probability score (lower is better); gives partial credit for near misses, e.g. D when the result is H."""
    onehot = np.eye(3)[y]
    cum_p, cum_o = np.cumsum(probs, axis=1), np.cumsum(onehot, axis=1)
    return np.mean(np.sum((cum_p - cum_o)[:, :2] ** 2, axis=1) / 2)


def score(probs_by_model, y):
    rows = []
    for name, p in probs_by_model.items():
        p = np.clip(p, 1e-6, 1)
        p = p / p.sum(axis=1, keepdims=True)
        pred = p.argmax(axis=1)
        rows.append({
            "model": name,
            "correct": int((pred == y).sum()),
            "n": len(y),
            "accuracy": accuracy_score(y, pred),
            "log_loss": log_loss(y, p, labels=[0, 1, 2]),
            "rps": rps(p, y),
            "draws_predicted": int((pred == 1).sum()),
        })
    return pd.DataFrame(rows).set_index("model")


def per_match_table(test_df, probs_by_model):
    out = test_df[["week", "date", "home_name", "away_name", "home_score", "away_score", "result"]].copy()
    for name, p in probs_by_model.items():
        short = name.split(" (")[0]
        out[short] = [OUTCOMES[i] for i in p.argmax(axis=1)]
    rf = probs_by_model["Random Forest"]
    out["RF P(H/D/A)"] = [f"{a:.2f}/{b:.2f}/{c:.2f}" for a, b, c in rf]
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int)
    ap.add_argument("--rolling", action="store_true")
    args = ap.parse_args()

    df = load_all_rows()
    finished_weeks = sorted(df.loc[df["season"] == CURRENT_SEASON, "week"].unique())
    # only gameweeks with all 10 matches finished
    counts = df[df["season"] == CURRENT_SEASON].groupby("week").size()
    full_weeks = [w for w in finished_weeks if counts[w] == 10]
    print(f"{CURRENT_SEASON}: finished gameweeks on disk = {list(map(int, full_weeks))}")

    weeks = full_weeks if args.rolling else [args.week or full_weeks[-1]]
    pooled_probs, pooled_y, tables = {}, [], []
    for week in weeks:
        train_df, test_df = split(df, week)
        y = test_df["result"].map(LABEL_MAP).to_numpy()
        probs = predict_all(train_df, test_df)
        print(f"\n=== Holdout: {CURRENT_SEASON} GW{week}  (train rows={len(train_df)}, test rows={len(test_df)}) ===")
        print(score(probs, y).round(3).to_string())
        tables.append(per_match_table(test_df, probs))
        for k, v in probs.items():
            pooled_probs.setdefault(k, []).append(v)
        pooled_y.append(y)

    if args.rolling:
        y = np.concatenate(pooled_y)
        probs = {k: np.vstack(v) for k, v in pooled_probs.items()}
        print(f"\n=== Pooled across GW{weeks[0]}-GW{weeks[-1]} ({len(y)} matches) ===")
        print(score(probs, y).round(3).to_string())
        print("\nActual results:", dict(pd.Series(y).map({0: "H", 1: "D", 2: "A"}).value_counts()))

    table = pd.concat(tables, ignore_index=True)
    print("\n=== Per-match predictions ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    main()
