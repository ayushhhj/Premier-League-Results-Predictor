import pandas as pd
from features import build_training_set, SEASONS

# Reminder: split must be chronological, not random (see docs/DECISIONS.md) — train on
# older seasons, evaluate on a more recent one, to mimic how the model actually gets used.
# SEASONS[0] is the current, still in-progress season — it has no complete season of its
# own results yet, so it's not a natural holdout candidate; probably either included in
# training as-is, or excluded until it's finished. Worth deciding before writing the split.


def load_training_data(seasons=SEASONS):
    """Build the full labeled training DataFrame via features.build_training_set()."""
    pass


def chronological_split(df, test_season):
    """Split df into train/test sets by season rather than randomly.

    test_season should be one of the completed past seasons (not the in-progress
    current one) — everything from that season becomes the held-out test set, and
    everything from older seasons becomes the training set.
    """
    pass


def select_features(df):
    """Pick out the feature columns to actually feed the model, separate from the label
    and from identifier columns (match_id, date, home_team, away_team) that shouldn't be
    used as raw inputs as-is. Decide here whether team identity gets encoded as a feature
    at all, or whether the derived stats (form, points, goal diff, h2h) are enough alone."""
    pass


def train_model(train_df):
    """Fit a Random Forest / XGBoost classifier on the training features against the
    result label (H/D/A)."""
    pass


def evaluate_model(model, test_df):
    """Evaluate the trained model against the held-out test set — accuracy at minimum,
    plus a confusion matrix or per-class breakdown to see if it's just predicting the
    majority class (home win) most of the time."""
    pass


def save_model(model, path):
    """Persist the trained model to disk under models/."""
    pass


if __name__ == "__main__":
    pass
