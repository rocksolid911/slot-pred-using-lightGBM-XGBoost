"""The demo page runs the exported trees itself, so the export must reproduce LightGBM exactly.

    .venv/Scripts/python.exe -m pytest tests/test_final.py -v
"""
import numpy as np
import pandas as pd

from slotpred.features import PrepState
from slotpred.final import export_trees, export_xgb_trees
from slotpred.train_lgbm import make_model
from slotpred.train_xgb import make_model as make_xgb


def _walk(n, x):
    """The same walk as the demo page's JavaScript."""
    while "v" not in n:
        v = x[n["f"]]
        if "c" in n:
            n = n["l"] if (v >= 0 and int(v) in n["c"]) else n["r"]
        else:
            n = n["l"] if v <= n["t"] else n["r"]
    return n["v"]


def test_exported_trees_match_lightgbm():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({"kind": pd.Categorical(rng.choice(["a", "b", "c", "d"], n), categories=["a", "b", "c", "d"]),
                      "reports": rng.integers(0, 8, n).astype(float), "hour": rng.integers(9, 18, n).astype(float)})
    y = (np.select([X["kind"] == "a", X["kind"] == "b"], [10.0, 20.0], 30.0) + 1.3 * X["reports"]) * rng.lognormal(-0.04, 0.28, n)
    m = make_model(0.8, n_estimators=60, min_child_samples=20).fit(X, y)
    trees = export_trees(m, PrepState(age_fill=0.0, categories={"kind": ["a", "b", "c", "d"]}))
    M = np.column_stack([X["kind"].cat.codes.to_numpy(), X["reports"], X["hour"]])
    mine = np.array([sum(_walk(t, row) for t in trees) for row in M[:300]])
    assert np.max(np.abs(mine - m.predict(X.iloc[:300]))) < 1e-9


def _walk_xgb(n, x):
    """The demo page's walk for XGBoost nodes ("lt" = strictly less than)."""
    while "v" not in n:
        v = x[n["f"]]
        if "c" in n:
            n = n["l"] if (v >= 0 and int(v) in n["c"]) else n["r"]
        else:
            n = n["l"] if v < n["lt"] else n["r"]
    return n["v"]


def test_exported_xgb_trees_match_xgboost():
    rng = np.random.default_rng(1)
    n = 3000
    X = pd.DataFrame({"kind": pd.Categorical(rng.choice(["a", "b", "c", "d"], n), categories=["a", "b", "c", "d"]),
                      "reports": rng.integers(0, 8, n).astype(float), "hour": rng.integers(9, 18, n).astype(float)})
    y = (np.select([X["kind"] == "a", X["kind"] == "b"], [10.0, 20.0], 30.0) + 1.3 * X["reports"]) * rng.lognormal(-0.04, 0.28, n)
    m = make_xgb(0.8, n_estimators=60, early_stopping_rounds=None, min_child_weight=10).fit(X, y)
    ex = export_xgb_trees(m)
    M = np.column_stack([X["kind"].cat.codes.to_numpy(), X["reports"], X["hour"]])
    mine = np.array([ex["base"] + sum(_walk_xgb(t, row) for t in ex["trees"]) for row in M[:300]])
    assert np.max(np.abs(mine - m.predict(X.iloc[:300]))) < 1e-3     # float32 sums, leaves rounded to 5 decimals
