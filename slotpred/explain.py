"""Lesson A9, part 1: why does the model say what it says?

Every prediction = a starting point + one PUSH per column (SHAP values (SHapley Additive exPlanations)).
 The pushes add up exactly to the prediction. Because we wrote the simulator, we can check a push against the hidden rule.

    P = pushes(booster, X)              one row per visit, one column per feature, plus "start"
    video_push_by_rule(7.0, 0.8)        the true push of a video consult for a 7-minute in-person visit

    .venv/Scripts/python.exe -m pytest tests/test_explain.py -v
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import DATA
from .features import fit_prep, make_xy
from .final import FINAL
from .prepare import load_prepared
from .world import DRIFT_DATE, DRIFT_DEPT, DRIFT_EXTRA_MIN, NOISE_SIGMA, TELE_FACTOR


# ---------------------------------------------------------------- TODO(you)

def pushes(booster, X: pd.DataFrame) -> pd.DataFrame:
    """Split each prediction into a starting point plus one push per column.

    Return a DataFrame with the same index as X and these columns, in this order:
        every column of X     that column's push for this visit, in minutes (+ means longer)
        "start"               the starting point (the same number on every row)
    so that each row adds up to booster.predict(X) for that visit.

    Hint: booster.predict(X, pred_contrib=True) returns a NumPy array with one row per visit and
    len(X.columns) + 1 columns: one per column of X, then the starting point last.
    """
    arr = booster.predict(X, pred_contrib=True)
    cols = list(X.columns) + ["start"]
    return pd.DataFrame(arr, index=X.index, columns=cols)


def video_push_by_rule(in_person_minutes, q, sigma=NOISE_SIGMA):
    """The TRUE push of a video consult on the q-point, from the hidden rule (world.py).

        in_person_minutes   the expected minutes this same visit would take in person (A2's
                            "work x pace", before drift); a number or a NumPy array
        q                   the P value, e.g. 0.8 (the PO's setting, A1)

    Two facts from A2:
        video multiplies the expected minutes by TELE_FACTOR (0.80)
        the q-point of a visit = expected minutes x e^(-sigma^2/2 + z_q * sigma), z_q = norm.ppf(q)

    So the push = (q-point by video) - (q-point in person). It is negative: video is shorter.
    Example: 7.0 minutes in person, q = 0.8 -> about -1.70.
    """
    z_q = norm.ppf(q)
    in_person_q = in_person_minutes * np.exp(-sigma**2 / 2 + z_q * sigma)
    video_q = in_person_q * TELE_FACTOR
    return video_q - in_person_q


# ---------------------------------------------------------------- Claude

def final_model(q):
    """The final LightGBM model for q (refitted on January-June in A8)."""
    return lgb.Booster(model_file=str(FINAL / f"lgbm_q{q:.2f}.txt"))


def test_table():
    """(X, visits) for the July-August test months, prepared exactly as in A8.

    `visits` holds the test rows plus two columns from the simulator's answer key (A2), which we may
    read here because we only explain the model, we never train on them:
        expected_minutes    the hidden rule's expected minutes for the visit
        in_person_minutes   the same visit in person, before the July Dermatology drift
    """
    splits, _ = load_prepared()
    state = fit_prep(pd.concat([splits["train"], splits["valid"]], ignore_index=True))
    test = splits["test"].reset_index(drop=True)
    X, _ = make_xy(test, state)
    truth = pd.read_csv(DATA / "_hidden_truth.csv")
    visits = test.merge(truth, on="appointment_id", how="left")
    drift = np.where((visits["department"] == DRIFT_DEPT) & (visits["date"] >= DRIFT_DATE), DRIFT_EXTRA_MIN, 0.0)
    video = np.where(visits["teleconsult"].astype(bool), TELE_FACTOR, 1.0)
    visits["in_person_minutes"] = (visits["expected_minutes"] - drift) / video
    return X, visits


def average_push(P: pd.DataFrame) -> pd.Series:
    """How much each column moves predictions on average (size only, ignoring up or down), largest first."""
    return P.drop(columns="start").abs().mean().sort_values(ascending=False)
