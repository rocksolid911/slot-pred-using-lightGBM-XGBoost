"""Lesson A5: simple predictions with no ML. They set the score LightGBM must beat (A6)
and become the fallback when the model service is down (A11).

Every baseline has the same two methods, like every model later:
    fit(train)     learn numbers from the TRAINING rows only (A4: a learned step)
    predict(df)    return one prediction per row of df, as a numpy array in df's row order

    B0  FixedSlot(15)                       today's rule: 15 minutes for everyone
    B1  GlobalQuantile(q)                   one number: the q-point of all training visits
    B2  GroupQuantile(keys, q)              one number per box, e.g. per department + visit type
    B3  GroupQuantile(finer keys, q, min_n, parent=B2)   smaller boxes, with backoff

    .venv/Scripts/python.exe -m pytest tests/test_baselines.py -v
"""
import numpy as np
import pandas as pd

TARGET = "minutes"


class FixedSlot:
    """B0: the same number of minutes for every visit. Learns nothing."""

    def __init__(self, minutes=15.0):
        self.minutes = minutes

    def fit(self, train):
        return self

    def predict(self, df):
        return np.full(len(df), float(self.minutes))


class GlobalQuantile:
    """B1: the q-point of ALL training visits, the same for everyone. Learns one number.

    A template for GroupQuantile: fit stores what it learned on self, predict uses it.
    """

    def __init__(self, q):
        self.q = q

    def fit(self, train):
        self.value_ = float(train[TARGET].quantile(self.q))
        return self

    def predict(self, df):
        return np.full(len(df), self.value_)


class GroupQuantile:
    """B2 / B3: one q-point per box. A box = one combination of the `keys` columns,
    e.g. keys=["department", "visit_type"] gives boxes like (Dermatology, follow-up).

    Backoff: a box with fewer than `min_n` training visits, or a box never seen in training
    (a new doctor), is not trusted. Its rows get the `parent` baseline's prediction instead.
    With parent=None the fallback is the q-point of all training visits (like B1).
    """

    def __init__(self, keys, q, min_n=1, parent=None):
        self.keys = list(keys)
        self.q = q
        self.min_n = min_n
        self.parent = parent

    def fit(self, train):
        """TODO(you). Learn from `train` and store on self:

            self.table_      the q-point of `minutes` for each box
                             hint: train.groupby(self.keys)[TARGET].quantile(self.q)
            self.counts_     how many training visits each box has     hint: .size()
            self.fallback_   the q-point of all training visits (used when parent is None)

        If self.parent is not None, fit it on the same `train` too.
        Return self.
        """
        self.table_ = train.groupby(self.keys)[TARGET].quantile(self.q)
        self.counts_ = train.groupby(self.keys).size()
        self.fallback_ = float(train[TARGET].quantile(self.q))
        if self.parent is not None:
            self.parent.fit(train)
        return self

    def predict(self, df):
        """TODO(you). One prediction per row of df, a numpy array in df's row order.

        1. Look up each row's box in self.table_ and self.counts_.
           hint: df.join(self.table_.rename("p"), on=self.keys) keeps df's rows and order
        2. A row is trusted if its box exists AND has at least self.min_n training visits.
        3. Untrusted rows get self.parent.predict(...) if there is a parent, else self.fallback_.
           (simplest: compute the parent's prediction for all rows, then pick with np.where)
        """
        parent_pred = self.parent.predict(df) if self.parent is not None else self.fallback_
        joined = df.join(self.table_.rename("p"), on=self.keys)
        counts = df.join(self.counts_.rename("n"), on=self.keys)
        trusted = counts["n"] >= self.min_n
        return np.where(trusted, joined["p"], parent_pred)


def oracle_quantile(expected, q, sigma):
    """The best possible P-point: needs the simulator's hidden expected minutes, so it is only
    for scoring (the floor no model can beat), never a feature. From A2: actual = expected x noise,
    and the noise's q-point is exp(-sigma^2/2 + z_q x sigma)."""
    from scipy.stats import norm
    return np.asarray(expected, float) * np.exp(-sigma ** 2 / 2 + norm.ppf(q) * sigma)


def as_frame(rows):
    """Leaderboard rows (dicts from metrics.evaluate) to a rounded table."""
    return pd.DataFrame(rows).round(3)
