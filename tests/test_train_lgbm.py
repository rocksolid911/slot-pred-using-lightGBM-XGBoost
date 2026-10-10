"""Checks for your make_model and fit_one in slotpred/train_lgbm.py.

    .venv/Scripts/python.exe -m pytest tests/test_train_lgbm.py -v
"""
import numpy as np
import pandas as pd
import pytest

from slotpred import train_lgbm as t
from slotpred.metrics import coverage, pinball_loss


def test_make_model_learns_the_q_point_with_our_settings():
    m = t.make_model(0.75)
    p = m.get_params()
    assert p["objective"] == "quantile" and p["alpha"] == 0.75
    assert p["learning_rate"] == t.PARAMS["learning_rate"] and p["num_leaves"] == t.PARAMS["num_leaves"]


def test_make_model_overrides():
    p = t.make_model(0.8, learning_rate=0.2, num_leaves=7).get_params()
    assert p["learning_rate"] == 0.2 and p["num_leaves"] == 7
    assert t.PARAMS["learning_rate"] == 0.05            # the shared settings are not changed


def _data(n, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"slow": rng.random(n) < 0.5, "reports": rng.integers(0, 6, n)}).astype(float)
    y = (np.where(X["slow"] == 1, 30.0, 10.0) + 1.3 * X["reports"]) * rng.lognormal(-0.04, 0.28, n)
    return X, pd.Series(y)


def test_fit_one_watches_valid_and_stops_early():
    Xtr, ytr = _data(2000, 0)
    Xva, yva = _data(500, 1)
    m = t.fit_one(0.8, Xtr, ytr, Xva, yva, patience=20, n_estimators=500, min_child_samples=10)
    assert 0 < m.best_iteration_ < 500                   # stopped before the maximum
    assert set(m.evals_result_) == {"train", "valid"}    # both scores recorded after every tree
    pred = m.predict(Xva)
    assert pinball_loss(yva, pred, 0.8) < pinball_loss(yva, np.full(len(yva), ytr.quantile(0.8)), 0.8)
    assert coverage(yva, pred) == pytest.approx(0.8, abs=0.06)
