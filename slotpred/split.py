"""Lesson A4, pipeline step 6: split by time (notes/A04-prepare.md, section 3).

    train       SIM_START   .. TRAIN_END     fit the model and every learned prep step
    valid       VALID_START .. VALID_END     choose settings, early stopping (used many times)
    test        TEST_START  .. SIM_END       the final honest score (used ONCE, in A8)

The dates live in slotpred/config.py; never type them here.

    .venv/Scripts/python.exe -m pytest tests/test_split.py -v
"""
import numpy as np
import pandas as pd

from .config import SIM_END, SIM_START, TEST_START, TRAIN_END, VALID_END, VALID_START

SPLITS = ("train", "valid", "test")


def time_split(df: pd.DataFrame, date_col="date") -> dict:
    """Return {"train": ..., "valid": ..., "test": ...}, three DataFrames chosen by `date_col`.

    Rules
        - Use the config dates; each range includes both of its end dates.
        - Every row lands in exactly one split.
        - Return copies, so changing a split later can't change `df`.

    `date_col` holds text "YYYY-MM-DD", which compares correctly as a string (or use .between).
    """
    # TODO(you)
    raise NotImplementedError


def random_split(df: pd.DataFrame, test_frac=0.2, seed=0) -> dict:
    """The WRONG split for this problem, kept only for the A4 experiment: rows shuffled across all months."""
    is_test = np.random.default_rng(seed).random(len(df)) < test_frac
    return {"train": df[~is_test].copy(), "test": df[is_test].copy()}


def describe(splits: dict, date_col="date") -> pd.DataFrame:
    """Rows and first/last date per split: the first thing to look at after splitting."""
    return pd.DataFrame({name: {"rows": len(s), "first": s[date_col].min(), "last": s[date_col].max()}
                         for name, s in splits.items()}).T
