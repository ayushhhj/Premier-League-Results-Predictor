import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import ParameterGrid
from statsmodels.miscmodels.ordinal_model import OrderedModel
import penaltyblog as pb  # Dixon-Coles
import joblib
from features import build_training_set, SEASONS

# Completed seasons, oldest first
COMPLETED_SEASONS_CHRONOLOGICAL = list(reversed(SEASONS[1:]))


def load_training_data(seasons=None):
    """Labelled rows for the given seasons; defaults to every completed season."""
    if seasons is None:
        seasons = SEASONS[1:]
    return build_training_set(seasons)


def chronological_split(df, test_season):
    """Split by season: test_season is held out and every other season in df is training data."""
    test_df = df[df["season"] == test_season]
    train_df = df[df["season"] != test_season]
    return train_df, test_df


def compute_season_weights(df, decay=0.8):
    """Per-row sample weights that decay with season age: decay ** (seasons before the newest in df).

    Not used by default; see docs/DECISIONS.md.
    """
    seasons_present = set(df["season"].unique())
    ordered = [s for s in COMPLETED_SEASONS_CHRONOLOGICAL if s in seasons_present]  # oldest -> newest
    seasons_ago = {season: len(ordered) - 1 - i for i, season in enumerate(ordered)}
    return df["season"].map(lambda season: decay ** seasons_ago[season])


# Team identity is deliberately left out; see docs/DECISIONS.md
FEATURE_COLUMNS = [
    "home_form", "away_form",
    "home_venue_form", "away_venue_form",
    "home_venue_points", "away_venue_points",
    "home_venue_goal_diff", "away_venue_goal_diff",
    "h2h_score", "home_rest_days", "away_rest_days",
    # home_xg/away_xg removed 2026-11-05; see docs/DECISIONS.md
]
LABEL_COLUMN = "result"
LABEL_MAP = {"H": 0, "D": 1, "A": 2}  # sklearn wants numeric labels, not strings


def get_features_and_labels(df):
    """(X, y) with FEATURE_COLUMNS and LABEL_MAP-encoded labels."""
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN].map(LABEL_MAP)
    return X, y

# Ordered labels for ordinal models: away win < draw < home win
ORDINAL_LABEL_MAP = {"A": 0, "D": 1, "H": 2}


def get_ordinal_features_and_labels(df):
    """(X, y) with ORDINAL_LABEL_MAP-encoded labels."""
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN].map(ORDINAL_LABEL_MAP)
    return X, y


def train_ordered_logit(train_df):
    """Fit an ordered logit (proportional odds) model on A < D < H.

    Predicted probabilities come out in [A, D, H] order.
    """
    X_train, y_train = get_ordinal_features_and_labels(train_df)
    model = OrderedModel(y_train, X_train, distr="logit")
    return model.fit(method="bfgs", disp=False)


def evaluate_ordered_logit(result, test_df):
    """(accuracy, confusion matrix, classification report) for a fitted ordered logit."""
    X_test, y_test = get_ordinal_features_and_labels(test_df)
    probs = result.model.predict(result.params, exog=X_test)
    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, cm, report


def train_lasso(train_df, C=1.0):
    """Fit an L1-regularised logistic regression on standardised features. Returns (model, scaler)."""
    X_train, y_train = get_features_and_labels(train_df)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(l1_ratio=1.0, solver = "saga", C=C, max_iter=5000)
    return model.fit(X_train_scaled, y_train), scaler

def evaluate_lasso(model, scaler, test_df):
    """(accuracy, confusion matrix, classification report), scaling test data with the training scaler."""
    X_test, y_test = get_features_and_labels(test_df)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, cm, report


def print_lasso_coefficients(model, name="Lasso"):
    """Print each class's coefficients, largest first, marking ones Lasso set to zero."""
    label_names = {v: k for k, v in LABEL_MAP.items()}
    print(f"\n=== {name} coefficients ===")
    for class_index, class_label in enumerate(model.classes_):
        pairs = sorted(
            zip(FEATURE_COLUMNS, model.coef_[class_index]),
            key=lambda pair: -abs(pair[1]),
        )
        print(f"\n-- class {label_names.get(class_label, class_label)} --")
        for feature, coef in pairs:
            marker = " (dropped)" if abs(coef) < 1e-4 else ""
            print(f"{feature}: {coef:.4f}{marker}")


def train_dixon_coles(train_df, half_life_days=180):
    """Fit a Dixon-Coles goals model (penaltyblog) on team identities and scores.

    Matches are weighted by recency with a half-life of half_life_days. Inputs are passed
    as numpy arrays because penaltyblog fails on pandas' read-only buffers. Models goal
    rates, so it's the starting point for scoreline prediction.
    """
    home_goals = np.array(train_df["home_score"].tolist())
    away_goals = np.array(train_df["away_score"].tolist())
    teams_home = np.array(train_df["home_team"].tolist())
    teams_away = np.array(train_df["away_team"].tolist())

    dates = pd.to_datetime(train_df["date"])
    days_ago = (dates.max() - dates).dt.days
    weights = np.array(0.5 ** (days_ago / half_life_days))

    model = pb.models.DixonColesGoalModel(home_goals, away_goals, teams_home, teams_away, weights=weights)
    model.fit()
    return model


def evaluate_dixon_coles(model, test_df, train_df):
    """(accuracy, confusion matrix, classification report) for Dixon-Coles.

    Skips matches involving a team that isn't in train_df (no ratings for it) and prints
    how many were skipped.
    """
    known_teams = set(train_df["home_team"]) | set(train_df["away_team"])

    y_true = []
    y_pred = []
    skipped = 0
    for _, row in test_df.iterrows():
        if row["home_team"] not in known_teams or row["away_team"] not in known_teams:
            skipped += 1
            continue
        result = model.predict(row["home_team"], row["away_team"])
        y_pred.append(int(np.argmax(result.home_draw_away)))  # 0=H, 1=D, 2=A
        y_true.append(LABEL_MAP[row["result"]])

    if skipped:
        print(f"(skipped {skipped}/{len(test_df)} rows — team unseen in training data)")

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return acc, cm, report


def train_baseline(train_df, strategy="most_frequent"):
    """DummyClassifier baseline; "most_frequent" always predicts the commonest outcome (home win)."""
    dummy = DummyClassifier(strategy=strategy)
    X_train, y_train = get_features_and_labels(train_df)
    return dummy.fit(X_train, y_train)


def train_random_forest(train_df, sample_weight=None):
    """Fit a RandomForestClassifier, optionally with per-row sample weights."""
    randomforest = RandomForestClassifier(n_estimators = 100, random_state = 12)
    X_train, y_train = get_features_and_labels(train_df)
    return randomforest.fit(X_train, y_train, sample_weight=sample_weight)


def evaluate_model(model, test_df):
    """(accuracy, confusion matrix, classification report) for a fitted sklearn model."""
    X_test, y_test = get_features_and_labels(test_df)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, cm, report


def walk_forward_splits(seasons=COMPLETED_SEASONS_CHRONOLOGICAL):
    """Yield (train_seasons, test_season) pairs, testing each season on only the seasons before it."""
    for i in range(1, len(seasons)):
        yield seasons[:i], seasons[i]


def tune_hyperparameters(model_class, param_grid, df, seasons=COMPLETED_SEASONS_CHRONOLOGICAL):
    """Grid search scored by mean walk-forward accuracy. Returns (best_params, best_score)."""
    best_params = None
    best_score = -1
    for params in ParameterGrid(param_grid):
        fold_scores = []
        for train_seasons, test_season in walk_forward_splits(seasons):
            train_df = df[df["season"].isin(train_seasons)]
            test_df = df[df["season"] == test_season]
            X_train, y_train = get_features_and_labels(train_df)
            X_test, y_test = get_features_and_labels(test_df)

            model = model_class(**params, random_state=12)
            model.fit(X_train, y_train)
            fold_scores.append(accuracy_score(y_test, model.predict(X_test)))

        avg_score = sum(fold_scores) / len(fold_scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    return best_params, best_score


RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_leaf": [1, 5, 10],
}

def print_feature_importances(model, name):
    """Print the model's feature importances, highest first."""
    importances = sorted(zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda pair: -pair[1])
    print(f"\n=== {name} feature importances ===")
    for feature, importance in importances:
        print(f"{feature}: {importance:.4f}")


def find_correlated_features(df, threshold=0.7):
    """Pairs of FEATURE_COLUMNS with |correlation| >= threshold, strongest first."""
    corr = df[FEATURE_COLUMNS].corr()
    pairs = []
    for i, col_a in enumerate(FEATURE_COLUMNS):
        for col_b in FEATURE_COLUMNS[i + 1:]:
            value = corr.loc[col_a, col_b]
            if abs(value) >= threshold:
                pairs.append((col_a, col_b, value))
    return sorted(pairs, key=lambda pair: -abs(pair[2]))


def save_model(model, path):
    """Save a fitted model with joblib."""
    joblib.dump(model, path)


if __name__ == "__main__":
    df = load_training_data()
    train_df, test_df = chronological_split(df, "2025/2026")

    print("=== Correlated feature pairs (|r| >= 0.7) ===")
    for col_a, col_b, value in find_correlated_features(train_df):
        print(f"{col_a} <-> {col_b}: {value:.3f}")

    baseline = train_baseline(train_df)
    rf = train_random_forest(train_df)

    for name, model in [("Baseline", baseline), ("Random Forest", rf)]:
        acc, cm, report = evaluate_model(model, test_df)
        print(f"\n=== {name} ===")
        print("Accuracy:", acc)
        print("Confusion matrix:\n", cm)
        print(report)

    ordered_logit = train_ordered_logit(train_df)
    acc, cm, report = evaluate_ordered_logit(ordered_logit, test_df)
    print("\n=== Ordered Logistic Regression ===")
    print("Accuracy:", acc)
    print("Confusion matrix:\n", cm)
    print(report)

    lasso, scaler = train_lasso(train_df)
    acc, cm, report = evaluate_lasso(lasso, scaler, test_df)
    print("\n=== Lasso ===")
    print("Accuracy:", acc)
    print("Confusion matrix:\n", cm)
    print(report)

    dixon_coles = train_dixon_coles(train_df)
    acc, cm, report = evaluate_dixon_coles(dixon_coles, test_df, train_df)
    print("\n=== Dixon-Coles ===")
    print("Accuracy:", acc)
    print("Confusion matrix:\n", cm)
    print(report)

    tuning_seasons = COMPLETED_SEASONS_CHRONOLOGICAL[:-1]

    best_rf_params, rf_cv_score = tune_hyperparameters(RandomForestClassifier, RF_PARAM_GRID, df, tuning_seasons)
    print(f"\nBest RF params: {best_rf_params} (walk-forward CV accuracy: {rf_cv_score:.4f})")

    X_train, y_train = get_features_and_labels(train_df)
    tuned_rf = RandomForestClassifier(**best_rf_params, random_state=12).fit(X_train, y_train)

    for name, model in [("Tuned Random Forest", tuned_rf)]:
        acc, cm, report = evaluate_model(model, test_df)
        print(f"\n=== {name} (final holdout: 2025/2026) ===")
        print("Accuracy:", acc)
        print("Confusion matrix:\n", cm)
        print(report)

    print_feature_importances(tuned_rf, "Tuned Random Forest")
