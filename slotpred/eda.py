"""Small plotting helpers for exploring the EMR export (lesson A3). They only look; they never change data."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype

from .config import DATA


def first_look(df: pd.DataFrame, date_col=None, max_levels=30):
    """A first pass over ANY table (notes/00-eda-playbook.md, steps 2-4).

    Prints structure and duplicates, missing % and distinct counts per column, numeric ranges
    (placeholders like 999 show up as the max), the values of low-cardinality text columns
    (spelling variants show up here) and, if `date_col` is given, rows per month (gaps, volume jumps).
    """
    print(f"{len(df):,} rows x {df.shape[1]} columns | exact duplicate rows: {df.duplicated().sum():,}")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_%": (df.isna().mean() * 100).round(2),
        "distinct": df.nunique(),
        "example": df.iloc[0],
    })
    print(summary.to_string())
    numeric = [c for c in df.columns if is_numeric_dtype(df[c]) and df[c].dtype != bool]
    if numeric:
        print(df[numeric].describe(percentiles=[0.01, 0.5, 0.99]).T.round(2).to_string())
    for c in df.columns:
        if is_string_dtype(df[c]) and df[c].nunique() <= max_levels:
            print(f"\n{c}:", df[c].value_counts(dropna=False).to_dict())
    if date_col:
        per_month = pd.to_datetime(df[date_col]).dt.to_period("M").value_counts().sort_index()
        print("\nrows per month:", per_month.rename(index=str).to_dict())


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
