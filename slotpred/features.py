"""Lesson A4, pipeline steps 7-8: LEARNED prep steps and the model's input table.

    state = fit_prep(train)              learn numbers from the TRAINING rows only
    X = make_features(any_split, state)  apply them, unchanged, to train / valid / test / API requests

fit_prep is the only function that looks at many rows at once. If it ever sees valid or test rows,
the scores in A5-A8 become optimistic (notes/A04-prepare.md, section 4).

    .venv/Scripts/python.exe -m pytest tests/test_features.py -v
"""
from dataclasses import dataclass

import pandas as pd

# Known at booking time (notes/A04-prepare.md, section 5). Everything else stays out of the model.
CATEGORICAL = ["department", "visit_type", "doctor", "clinic"]
NUMERIC = ["age", "age_missing", "interpreter", "teleconsult", "prior_reports", "weekday", "hour"]
FEATURES = CATEGORICAL + NUMERIC
FORBIDDEN = {"appointment_id", "patient_id", "patient_name", "date", "slot_time",
             "check_in", "consult_start", "consult_end", "minutes"}
TARGET = "minutes"


@dataclass
class PrepState:
    """Everything fit_prep learned. In A11 the API loads this next to the model."""
    age_fill: float                 # median age of the training rows
    categories: dict                # column -> sorted list of the levels seen in training


# ---------------------------------------------------------------- TODO(you)

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with two new int columns:
        weekday   0 = Monday .. 6 = Sunday, from `date` ("YYYY-MM-DD")
        hour      9 .. 17, from `slot_time` ("HH:MM")
    This is a RULE (row by row), so it runs the same way everywhere.
    """
    # TODO(you)
    raise NotImplementedError


def impute_age(age: pd.Series, fill: float):
    """Return (age_filled, age_missing).

        age_filled   float: `age` with every NaN replaced by `fill`
        age_missing  int 0/1: 1 where `age` was NaN, so the model can tell
                     "really 42" apart from "unknown, filled with 42"
    """
    # TODO(you)
    raise NotImplementedError


def fit_prep(train: pd.DataFrame) -> PrepState:
    """Learn the prep numbers from the TRAINING rows only.

        age_fill     the median of train["age"] (NaN is ignored by pandas' median)
        categories   for each column in CATEGORICAL: the sorted list of distinct, non-missing values
    """
    # TODO(you)
    raise NotImplementedError


# ---------------------------------------------------------------- Claude

def make_features(df: pd.DataFrame, state: PrepState) -> pd.DataFrame:
    """The model's input table X: exactly the FEATURES columns, in that order.

    Categorical columns get a pandas category dtype with the TRAINING levels. LightGBM uses these
    natively (A6). A level never seen in training (a new doctor) becomes NaN instead of crashing,
    and the model treats it like a missing value.
    """
    out = add_time_features(df)
    out["age"], out["age_missing"] = impute_age(out["age"], state.age_fill)
    for col in CATEGORICAL:
        known = out[col].where(out[col].isin(state.categories[col]))     # unseen level -> NaN
        out[col] = pd.Categorical(known, categories=state.categories[col])
    for col in ("interpreter", "teleconsult"):
        out[col] = out[col].astype(int)
    return out[FEATURES]


def make_xy(df: pd.DataFrame, state: PrepState):
    """(X, y) for one split."""
    return make_features(df, state), df[TARGET].astype(float)
