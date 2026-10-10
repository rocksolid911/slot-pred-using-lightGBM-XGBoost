"""Lesson A7: XGBoost, a second boosting library, trained exactly like LightGBM (A6) for a fair comparison.

Same data (A4 features), same q values (the PO's settings), same June early stopping, same score.
The differences are inside the library:
    LightGBM grows each tree LEAF-WISE: always split the box that lowers the score most (num_leaves caps it)
    XGBoost  grows each tree LEVEL-WISE: split every box on one level, then the next (max_depth caps it)

    .venv/Scripts/python.exe -m pytest tests/test_train_xgb.py -v
"""
import json

import xgboost as xgb

from .config import OUT, Q_CANDIDATES, SEED

MODELS = OUT / "models"

PARAMS = dict(
    n_estimators=2000,          # maximum trees; early stopping picks fewer
    learning_rate=0.05,         # same step size as LightGBM
    max_depth=6,                # at most 6 levels of questions = up to 64 boxes per tree
    min_child_weight=30,        # for this objective: about 30 training visits per box (A5's backoff idea)
    tree_method="hist",         # bins numbers first, like LightGBM: fast
    enable_categorical=True,    # use our category columns (doctor, department...) directly
    early_stopping_rounds=100,  # stop when June hasn't improved for 100 trees
    random_state=SEED,
)


def make_model(q, **overrides):
    """An untrained XGBRegressor that learns the q-point. XGBoost calls q 'quantile_alpha'."""
    return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, **{**PARAMS, **overrides})


def fit_one(q, X_train, y_train, X_valid, y_valid, **overrides):
    """Train one model for one q, scoring train and June after every tree; early stopping watches
    the LAST set in eval_set (June). model.best_iteration is the best tree count (counting from 0)."""
    model = make_model(q, **overrides)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=False)
    return model


def fit_all(X_train, y_train, X_valid, y_valid, qs=Q_CANDIDATES, save=True):
    """One model per q. Returns {q: model}; with save=True also writes outputs/models/xgb_q<q>.json."""
    models = {q: fit_one(q, X_train, y_train, X_valid, y_valid) for q in qs}
    if save:
        MODELS.mkdir(parents=True, exist_ok=True)
        for q, m in models.items():
            m.save_model(MODELS / f"xgb_q{q:.2f}.json")
        (MODELS / "xgb_params.json").write_text(json.dumps(
            {"params": PARAMS, "best_iteration": {f"{q:.2f}": int(m.best_iteration) + 1 for q, m in models.items()}}, indent=1))
    return models
