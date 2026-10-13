"""Lesson A9, part 2: is the model fair to every group?

"86% on time" is an average. An average can hide a group that does much worse. So we cut the
booked results by groups that matter (patients' age, video, clinic, department, new doctor) and ask
two questions of each group:
    1. is it clearly below the overall on-time share, beyond what luck explains?
    2. is it better or worse off than with today's fixed 15 minutes?

    t = group_table(visits, "age_band")

    .venv/Scripts/python.exe -m pytest tests/test_fairness.py -v
"""
import numpy as np
import pandas as pd

from .baselines import GroupQuantile
from .explain import final_model, test_table
from .features import make_xy
from .metrics import to_slot
from .prepare import load_prepared
from .train_lgbm import load_booster

TODAY = 15.0            # today's rule: 15 minutes for everyone
GROUPS = ["age_band", "visit_type", "video", "clinic", "department", "new_doctor"]


# ---------------------------------------------------------------- TODO(you)

def luck_band(p, n):
    """How far a group's on-time share can wander from p by luck alone, with n visits.

    If every visit is on time with chance p, the share in a group of n visits jumps around p with a
    standard deviation of sqrt(p * (1 - p) / n) (A5: the same 1/sqrt(n) idea as a box's P80).
    Return TWO standard deviations: about 95% of lucky wobbles stay inside that band.

    Example: p = 0.86, n = 461 -> about 0.032 (3.2 percentage points).
    Works for numbers and for NumPy arrays / pandas Series of n.
    """
    raise NotImplementedError("TODO(you): A9 part 2")


def group_table(visits: pd.DataFrame, group: str) -> pd.DataFrame:
    """One row per level of `group` (e.g. "65+", "adult 16-64", ...), sorted by on_time, lowest first.

    `visits` has one row per visit with at least these columns:
        group          the column named by `group`
        on_time        True if the visit fitted in the slot the model booked
        late_min       minutes past the booked slot (0 if it fitted)
        today_on_time  True if the visit fitted in today's 15 minutes

    Columns to return:
        n              number of visits in the group
        on_time        share on time with the model
        today          share on time with today's 15 minutes
        late_min       average late minutes per visit
        gap            on_time - (on-time share of ALL visits)       negative = worse than average
        band           luck_band(on-time share of ALL visits, n)
        flag           True when the group is below average by more than luck: gap < -band
    """
    raise NotImplementedError("TODO(you): A9 part 2")


# ---------------------------------------------------------------- Claude

def _groups(df, unseen):
    age = df["age"]
    band = pd.cut(age, [-1, 2, 15, 64, 200], labels=["baby 0-2", "child 3-15", "adult 16-64", "65+"]).astype(str)
    return df.assign(age_band=band.where(age.notna(), "unknown"),
                     video=np.where(df["teleconsult"].astype(bool), "video", "in person"),
                     new_doctor=np.where(unseen, "new doctor", "known doctor"))


def _booked(df, pred):
    """Adds pred (the model's q-point), slot (booked), covered (fits the q-point before rounding),
    on_time (fits the booked slot), late_min and today_on_time."""
    y = df["minutes"].to_numpy()
    slot = to_slot(pred)
    return df.assign(pred=pred, slot=slot, covered=y <= pred, on_time=y <= slot,
                     late_min=np.maximum(y - slot, 0), today_on_time=y <= TODAY)



def booked_test(q):
    """July-August with the FINAL model from A8 (Jan-Jun, + the new-doctor fallback): one row per visit."""
    X, v = test_table()
    splits, _ = load_prepared()
    train_valid = pd.concat([splits["train"], splits["valid"]], ignore_index=True)
    unseen = ~v["doctor"].isin(train_valid["doctor"].unique()).to_numpy()
    pred = final_model(q).predict(X)
    pred = np.where(unseen, GroupQuantile(["department", "visit_type"], q).fit(train_valid).predict(v), pred)
    return _booked(_groups(v, unseen), pred)


def booked_june(q):
    """June with the A6 model (trained Jan-May): before the July changes, so drift can't explain a gap here."""
    splits, state = load_prepared()
    v = splits["valid"].reset_index(drop=True)
    X, _ = make_xy(v, state)
    return _booked(_groups(v, np.zeros(len(v), bool)), load_booster(q).predict(X))
