"""Lesson A8: the final, honest evaluation, run ONCE on the locked test months (July-August).

    .venv/Scripts/python.exe -m slotpred.final

What it does
    1. Refit the chosen model (LightGBM, A7) on train + valid (Jan-Jun). The number of trees is the one
       early stopping picked on June (A6), so nothing is chosen by looking at the test months.
       Allowed and normal (A4 lab 4.3): valid was for choosing settings; once chosen, its rows can train.
    2. Refit the A5 baselines on Jan-Jun too, so the comparison is fair.
    3. Score everything on Jul-Aug: overall, by department (the Dermatology change from 1 July),
       for the new doctor (Nikhil Verma, never seen in training), and by month.
    4. Build the trade-off table the PO chooses q from: on-time share, late and idle minutes,
       average slot and patients per 4-hour session, for each q, next to today's 15-minute rule.

Writes outputs/final/test_results.json, the final models (outputs/final/lgbm_q<q>.txt) and
outputs/final/demo_model.json (the trees in a form the demo page can run).
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from .baselines import FixedSlot, GroupQuantile, oracle_quantile
from .config import DATA, OUT, Q_CANDIDATES, SLOT_GRID
from .features import CATEGORICAL, FEATURES, fit_prep, make_xy
from .metrics import evaluate, to_slot
from .prepare import load_prepared
from .train_lgbm import fit_one, make_model
from .world import NOISE_SIGMA

FINAL = OUT / "final"
SESSION_MIN = 240
# Extra safety levels for the demo page, so the PO can explore beyond the three candidates (A1).
# Each gets its own model, built the same way: tree count chosen on June, refit on Jan-Jun.
Q_DEMO = (0.50, 0.60, 0.67, 0.70, 0.75, 0.80, 0.85, 0.90)           # a 4-hour OPD session, to express slots as patients per session


def choose_trees(splits, state, qs=Q_DEMO):
    """A6 again: train on Jan-May, early-stop on June, return {q: number of trees}."""
    Xtr, ytr = make_xy(splits["train"], state)
    Xva, yva = make_xy(splits["valid"], state)
    return {q: int(fit_one(q, Xtr, ytr, Xva, yva).best_iteration_) for q in qs}


def refit(train_valid, state, trees):
    """Final models on Jan-Jun with the June-chosen number of trees (no early stopping, no test data)."""
    X, y = make_xy(train_valid, state)
    return {q: make_model(q, n_estimators=n).fit(X, y) for q, n in trees.items()}


def booked_summary(y, slot):
    """What the clinic feels: share on time, minutes late / idle per visit, slot length, patients per session."""
    y, slot = np.asarray(y, float), np.asarray(slot, float)
    return {"on_time": float(np.mean(y <= slot)), "late_min": float(np.mean(np.maximum(y - slot, 0))),
            "idle_min": float(np.mean(np.maximum(slot - y, 0))), "slot_min": float(np.mean(slot)),
            "patients_per_session": float(SESSION_MIN / np.mean(slot)), "n": int(len(y))}


def slices(test, y, preds):
    """Scores per group of test visits: department, new doctor vs known, month."""
    out = {}
    groups = {"department": test["department"].to_numpy(),
              "doctor_known": np.where(test["doctor"] == "Nikhil Verma", "new doctor (Nikhil Verma)", "known doctors"),
              "month": test["date"].str[:7].to_numpy()}
    for name, g in groups.items():
        out[name] = {}
        for level in sorted(set(g)):
            m = g == level
            out[name][level] = {method: {"bias": float(np.mean(y[m] - p[m])), **booked_summary(y[m], to_slot(p[m]))}
                                for method, p in preds.items()}
    return out


def export_trees(model, state):
    """The model's trees as plain JSON so a web page can evaluate them (demo). Category columns use the
    position of the level in state.categories, exactly the codes LightGBM saw in training."""
    dump = model.booster_.dump_model()

    def node(n):
        if "leaf_value" in n:
            return {"v": n["leaf_value"]}
        d = {"f": n["split_feature"], "l": node(n["left_child"]), "r": node(n["right_child"])}
        if n["decision_type"] == "==":
            d["c"] = [int(c) for c in str(n["threshold"]).split("||")]
        else:
            d["t"] = n["threshold"]
        return d

    return [node(t["tree_structure"]) for t in dump["tree_info"]]


def main():
    splits, _ = load_prepared()
    train_valid = pd.concat([splits["train"], splits["valid"]], ignore_index=True)
    test = splits["test"].reset_index(drop=True)
    state_tv = fit_prep(train_valid)              # learned prep steps refitted on Jan-Jun, like the model

    trees = choose_trees(splits, fit_prep(splits["train"]))
    models = refit(train_valid, state_tv, trees)
    X_test, y_test = make_xy(test, state_tv)
    y = y_test.to_numpy()

    truth = pd.read_csv(DATA / "_hidden_truth.csv")
    expected = test[["appointment_id"]].merge(truth, on="appointment_id", how="left")["expected_minutes"].to_numpy()

    board, tradeoff, by_slice = [], {}, {}
    today = FixedSlot(15).predict(test)
    tradeoff["today (fixed 15)"] = booked_summary(y, to_slot(today))
    for q in Q_CANDIDATES:
        b3 = GroupQuantile(["doctor", "visit_type"], q, min_n=30,
                           parent=GroupQuantile(["department", "visit_type"], q)).fit(train_valid)
        lgbm = models[q].predict(X_test)
        # A5's backoff for the model: a doctor the model never saw gets the lookup table's
        # department + visit type number instead (the model leans on `doctor`, so unknown doctors drift)
        unseen = ~test["doctor"].isin(state_tv.categories["doctor"]).to_numpy()
        hybrid = np.where(unseen, b3.predict(test), lgbm)
        preds = {"today (fixed 15)": today, "B3 lookup table": b3.predict(test), "LightGBM": lgbm,
                 "LightGBM + new-doctor backoff": hybrid, "Oracle (floor)": oracle_quantile(expected, q, NOISE_SIGMA)}
        board += [evaluate(y, p, q, name) for name, p in preds.items()]
        tradeoff[f"Model q={q:.2f}"] = booked_summary(y, to_slot(hybrid))
        by_slice[f"{q:.2f}"] = slices(test, y, {k: preds[k] for k in
                                                ("today (fixed 15)", "B3 lookup table", "LightGBM", "LightGBM + new-doctor backoff")})

    # every demo q on the test visits (model + new-doctor fallback), for the page's live trade-off table.
    # Models for different q are trained separately and can cross (a P85 below the P80 for some visit);
    # taking the running maximum over q keeps them in order, the standard fix ("rearrangement").
    unseen = ~test["doctor"].isin(state_tv.categories["doctor"]).to_numpy()
    lookup = {q: GroupQuantile(["department", "visit_type"], q).fit(train_valid).predict(test) for q in Q_DEMO}
    stack = np.maximum.accumulate(np.array([np.where(unseen, lookup[q], models[q].predict(X_test)) for q in Q_DEMO]), axis=0)
    demo_test = {"y": [round(float(v), 2) for v in y], "q": [f"{q:.2f}" for q in Q_DEMO],
                 "pred": {f"{q:.2f}": [round(float(v), 2) for v in row] for q, row in zip(Q_DEMO, stack)}}

    FINAL.mkdir(parents=True, exist_ok=True)
    for q, m in models.items():
        m.booster_.save_model(FINAL / f"lgbm_q{q:.2f}.txt")
    results = {"trees": {f"{q:.2f}": t for q, t in trees.items()}, "test_rows": int(len(test)),
               "train_rows": int(len(train_valid)), "board": board, "tradeoff": tradeoff, "slices": by_slice,
               "demo_test": demo_test}
    (FINAL / "test_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    backoff = {f"{q:.2f}": {f"{d}|{v}": round(float(x), 3) for (d, v), x in
                            GroupQuantile(["department", "visit_type"], q).fit(train_valid).table_.items()}
               for q in Q_DEMO}
    doctors = train_valid.groupby("doctor")["department"].first().to_dict()
    demo = {"features": FEATURES, "categorical": CATEGORICAL, "categories": state_tv.categories,
            "backoff": backoff, "doctor_department": doctors,
            "age_fill": state_tv.age_fill, "grid": SLOT_GRID,
            "models": {f"{q:.2f}": export_trees(m, state_tv) for q, m in models.items()}}
    (FINAL / "demo_model.json").write_text(json.dumps(demo, separators=(",", ":")), encoding="utf-8")

    show = pd.DataFrame(board)[["method", "q", "pinball", "coverage", "booked_on_time", "booked_idle_min", "booked_slot_min"]]
    print("trees chosen on June:", trees)
    print(show.round(3).to_string(index=False))
    print(pd.DataFrame(tradeoff).T.round(3).to_string())
    return results, models, state_tv


if __name__ == "__main__":
    main()
