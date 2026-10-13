"""Checks for your two TODO(you) functions in slotpred/explain.py.

    .venv/Scripts/python.exe -m pytest tests/test_explain.py -v
"""
import numpy as np
import pandas as pd
import pytest

from slotpred.explain import pushes, video_push_by_rule
from slotpred.train_lgbm import make_model
from slotpred.world import TELE_FACTOR


@pytest.fixture(scope="module")
def tiny():
    rng = np.random.default_rng(3)
    n = 2000
    X = pd.DataFrame({"kind": pd.Categorical(rng.choice(["a", "b", "c"], n), categories=["a", "b", "c"]),
                      "reports": rng.integers(0, 8, n).astype(float), "video": rng.integers(0, 2, n)},
                     index=pd.RangeIndex(100, 100 + n))
    y = (np.select([X["kind"] == "a", X["kind"] == "b"], [8.0, 15.0], 25.0) + 1.3 * X["reports"]) \
        * np.where(X["video"] == 1, 0.8, 1.0) * rng.lognormal(-0.04, 0.28, n)
    booster = make_model(0.8, n_estimators=40).fit(X, y).booster_
    return booster, X


# ---------------------------------------------------------------- pushes

def test_pushes_has_one_column_per_feature_plus_start(tiny):
    booster, X = tiny
    P = pushes(booster, X)
    assert list(P.columns) == list(X.columns) + ["start"]
    assert P.index.equals(X.index)


def test_pushes_add_up_to_the_prediction(tiny):
    booster, X = tiny
    P = pushes(booster, X)
    assert np.allclose(P.sum(axis=1).to_numpy(), booster.predict(X), atol=1e-9)


def test_start_is_the_same_for_every_visit(tiny):
    booster, X = tiny
    assert pushes(booster, X)["start"].nunique() == 1


# ---------------------------------------------------------------- video_push_by_rule

def test_video_push_worked_example():
    # 7 min in person -> P80 8.52; by video 5.6 min -> P80 6.82; push -1.70
    assert video_push_by_rule(7.0, 0.8) == pytest.approx(-1.703, abs=0.002)


def test_video_push_at_the_median():
    # q = 0.5: z = 0, so the q-point is expected x e^(-sigma^2/2)
    assert video_push_by_rule(7.0, 0.5) == pytest.approx(7.0 * (TELE_FACTOR - 1) * np.exp(-0.28**2 / 2), abs=1e-6)


def test_video_push_grows_with_the_visit_and_takes_arrays():
    out = video_push_by_rule(np.array([7.0, 14.0]), 0.8)
    assert out[1] == pytest.approx(2 * out[0])
    assert (out < 0).all()
