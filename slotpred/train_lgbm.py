"""Lesson A6, part 2: the real LightGBM, one model per q.

    q (alpha)            business setting from the PO (A1); config.Q_CANDIDATES
    everything in PARAMS our settings; tuned properly in A9

Fit on train (Jan-May). June (valid) is watched while training: early stopping keeps the number of trees
where June's score was best. The test months stay locked until A8.

    .venv/Scripts/python.exe -m pytest tests/test_train_lgbm.py -v
"""
import json

import lightgbm as lgb

from .config import OUT, Q_CANDIDATES, SEED

MODELS = OUT / "models"

PARAMS = dict(
    n_estimators=2000,        # maximum number of trees; early stopping picks fewer
    learning_rate=0.05,       # each tree moves the predictions 5% of the way
    num_leaves=31,            # at most 31 boxes per tree
    min_child_samples=30,     # a box needs at least 30 training visits (A5's backoff rule)
    random_state=SEED,
    verbose=-1,               # no training chatter
)


# ---------------------------------------------------------------- TODO(you)

def make_model(q, **overrides):
    """An untrained LGBMRegressor that learns the q-point.

        objective="quantile" and alpha=q   (LightGBM calls q 'alpha')
        plus every setting in PARAMS, where `overrides` replaces any of them
        e.g. make_model(0.8, learning_rate=0.1)

    hint: lgb.LGBMRegressor(objective=..., alpha=..., **{**PARAMS, **overrides})
    """
    return lgb.LGBMRegressor(objective="quantile", alpha=q, **{**PARAMS, **overrides})


def fit_one(q, X_train, y_train, X_valid, y_valid, patience=100, **overrides):
    """Train one model for one q, watching June. Return the fitted model.

        model = make_model(q, **overrides)
        model.fit(X_train, y_train,
                  eval_X=(X_train, X_valid), eval_y=(y_train, y_valid),   # score both after every tree
                  eval_names=["train", "valid"], eval_metric="quantile",
                  callbacks=[lgb.early_stopping(patience, verbose=False)])

    After fitting, model.best_iteration_ is the number of trees kept, and model.evals_result_ holds
    the train and valid score after every tree (for the curve in the notebook).
    LightGBM recognises X_train as the training data, so early stopping ignores it and watches June only.
    """
    model = make_model(q, **overrides)
    model.fit(X_train, y_train,
              eval_X=(X_train, X_valid), eval_y=(y_train, y_valid),   # score both after every tree
              eval_names=["train", "valid"], eval_metric="quantile",
              callbacks=[lgb.early_stopping(patience, verbose=False)])
    return model


# ---------------------------------------------------------------- Claude

def fit_all(X_train, y_train, X_valid, y_valid, qs=Q_CANDIDATES, save=True):
    """One model per q. Returns {q: model}; with save=True also writes outputs/models/lgbm_q<q>.txt."""
    models = {q: fit_one(q, X_train, y_train, X_valid, y_valid) for q in qs}
    if save:
        MODELS.mkdir(parents=True, exist_ok=True)
        for q, m in models.items():
            m.booster_.save_model(MODELS / f"lgbm_q{q:.2f}.txt", num_iteration=m.best_iteration_)
        (MODELS / "lgbm_params.json").write_text(json.dumps({"params": PARAMS, "best_iteration":
                                                            {f"{q:.2f}": m.best_iteration_ for q, m in models.items()}}, indent=1))
    return models


def load_booster(q):
    """A saved model for q, ready for booster.predict(X)."""
    return lgb.Booster(model_file=str(MODELS / f"lgbm_q{q:.2f}.txt"))
