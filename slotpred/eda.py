"""Small plotting helpers for exploring the EMR export (lesson A3). They only look; they never change data."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DATA


def load_raw() -> pd.DataFrame:
    """The EMR export exactly as the hospital hands it over: no parsing, no cleaning."""
    return pd.read_csv(DATA / "raw_emr_export.csv")


def log_hist(values, title="", ax=None, bins=80):
    """Histogram with a log-scaled x axis, so 0.5-minute and 300-minute values are both visible."""
    v = pd.Series(values).dropna()
    v = v[v > 0]
    ax = ax or plt.subplots(figsize=(9, 3.5))[1]
    ax.hist(v, bins=np.logspace(np.log10(v.min()), np.log10(v.max()), bins))
    ax.set_xscale("log")
    ax.set_xlabel("minutes (log scale)")
    ax.set_ylabel("rows")
    ax.set_title(title)
    return ax


def monthly_median(df, value_col, by, date_col="date", ax=None):
    """One line per group: the monthly median of `value_col`. Process changes show up as steps."""
    month = pd.to_datetime(df[date_col]).dt.to_period("M").dt.start_time
    table = df.assign(month=month).groupby(["month", by])[value_col].median().unstack()
    ax = ax or plt.subplots(figsize=(9, 3.5))[1]
    table.plot(ax=ax, marker="o")
    ax.set_ylabel(f"median {value_col}")
    ax.legend(fontsize=8, ncol=2)
    return table
