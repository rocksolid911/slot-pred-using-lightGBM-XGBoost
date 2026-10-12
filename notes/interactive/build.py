"""Build the interactive notes: one self-contained page per lesson, plus a hub page.

    .venv/Scripts/python.exe notes/interactive/build.py

Inputs   src/common.css, src/common.js   shared by every page
         src/<ID>.html                   one lesson's labs (markup + <script>)
         urls.json                       published artifact URL of each page (filled in after publishing)
Outputs  <slug>.html for every built lesson, and index.html (the hub)

The simulator's constants and July-August data are embedded in each page that needs them,
so rerun this after changing slotpred/world.py.
To add a lesson: write src/<ID>.html, set "src" on its entry in LESSONS, rebuild, publish.
"""
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SRC = HERE / "src"
URLS = HERE / "urls.json"
BIN = 0.5

# status: done | now | planned.  "src" marks lessons that have a lab page.
LESSONS = [
    dict(id="A1", slug="A01-framing", src="A01.html", status="done", data=False, title="A1 Framing Lab",
         name="Framing", stage="Stage 0 · Framing", note="notes/A01-framing.md",
         summary="Why the slot is a quantile, not the average: slot cost, pinball loss, and turning PO answers into q.",
         labs=["Slot cost", "Pinball loss", "Ask the PO"]),
    dict(id="A2", slug="A02-simulator", src="A02.html", status="done", data=True, title="A2 Simulator Lab",
         name="The simulated hospital", stage="Stage 1 · Collect", note="notes/A02-simulator.md",
         summary="actual = hidden rule × log-normal noise; the oracle floor; who a fixed 15-minute slot hurts.",
         labs=["Hidden rule", "Log-normal noise", "z-scores", "Oracle", "Heatmap"]),
    dict(id="A3", slug="A03-eda", src="A03.html", status="done", data=True, title="A3 EDA Lab",
         name="Exploring the messy export (EDA)", stage="Stage 2 · Explore", note="notes/A03-eda.md",
         summary="Robust statistics, outlier rules scored against the truth, MCAR vs MAR vs MNAR, and a process change no rule can see.",
         labs=["Robust statistics", "Outlier rules vs truth", "MCAR, MAR, MNAR", "The Whitefield step"]),
    dict(id="A4", slug="A04-prepare", src="A04.html", status="done", data=True, title="A4 Prepare Lab",
         name="Cleaning, features, time split", stage="Stage 3 · Prepare", note="notes/A04-prepare.md",
         summary="Rules run anywhere, learned steps are fitted on train only, and the split follows time.",
         labs=["Cleaning funnel", "Random vs time split", "What may the fit see?", "Rule or learned?"]),
    dict(id="A5", slug="A05-baselines", src="A05.html", status="done", data=True, title="A5 Baselines Lab",
         name="Baselines", stage="Stage 4 · Baseline", note="notes/A05-baselines.md",
         summary="The score to beat: pinball loss, four simple baselines on June, why tiny boxes are luck, and rounding to the grid.",
         labs=["The score", "June leaderboard", "Box size", "Rounding"]),
    dict(id="A6", slug="A06-lightgbm", src="A06.html", status="done", data=True, title="A6 LightGBM Lab",
         name="LightGBM quantile model", stage="Stage 5 · Train", note="notes/A06-lightgbm.md",
         summary="Boosting: many small trees, each nudging visits towards their q-point; early stopping on June; one model per q.",
         labs=["Boosting round by round", "Training curve", "LightGBM vs baselines"]),
    dict(id="A7", slug="A07-xgboost", src="A07.html", status="done", data=True, title="A7 XGBoost Lab",
         name="XGBoost comparison", stage="Stage 5 · Train", note="notes/A07-xgboost.md",
         summary="A second model trained the same way: leaf-by-leaf vs level-by-level trees, and a fair comparison on June.",
         labs=["Leaf vs level", "Fair comparison"]),
    dict(id="A8", slug="A08-final-test", src="A08.html", status="done", data=True, title="A8 Final Test Lab",
         name="The final test", stage="Stage 6 · Evaluate", note="notes/A08-final-test.md",
         summary="One honest look at July–August: why only once, the trade-off table for the PO, and the drift June could not show.",
         labs=["Why only once", "Trade-off table", "Drift and new doctor"]),
    dict(id="A9", status="now", name="Tuning, SHAP, fairness", stage="Stage 7 · Select", summary="Did the model find the hidden rule? Model card."),
    dict(id="A10", name="Experiment tracking", stage="Cross-cutting", summary="MLflow runs and model versions."),
    dict(id="A11", name="Serving", stage="Stage 8 · Deploy", summary="FastAPI, container, shadow mode and a fallback."),
    dict(id="A12", name="Monitoring", stage="Stage 9 · Monitor", summary="Catch the July drift: new doctor, Dermatology protocol."),
    dict(id="A13", name="Retraining", stage="Stage 10 · Retrain", summary="Champion vs challenger, rollback."),
    dict(id="B1", name="Pretraining a tiny GPT", stage="Track B · learning", summary="Tokenizer, attention, next-token loss, from scratch."),
    dict(id="B2", name="SFT and loss masking", stage="Track B · learning", summary="Why pretraining helps when labelled data is small."),
    dict(id="B3", name="Continued pretraining of Qwen", stage="Track B · learning", summary="LoRA with Unsloth on clinic text."),
    dict(id="B4", name="SFT on Qwen", stage="Track B · learning", summary="Sentence in, \"14 min\" out."),
    dict(id="B5", name="RL with GRPO", stage="Track B · learning", summary="A pinball-loss reward pushes the model to the safe quantile."),
    dict(id="B6", name="Extractor for LightGBM", stage="Track B · learning", summary="Sentence to JSON fields, feeding the production model."),
    dict(id="B7", name="Face-off", stage="Track B · learning", summary="LLM vs extractor + LightGBM vs LightGBM."),
]

HEAD = """<meta charset="utf-8">
<title>{{TITLE}}</title>
<meta name="description" content="{{DESC}}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=Spectral:wght@500;600&display=swap">
<style>
{{CSS}}
</style>
"""

LIBS = """<script>
window.MathJax = { tex: { inlineMath: [["\\\\(", "\\\\)"]], displayMath: [["\\\\[", "\\\\]"]] }, svg: { fontCache: "global" }, options: { enableMenu: false } };
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>window.SLOT_DATA = {{DATA}};</script>
<script>
{{COMMON_JS}}
</script>
"""


def slot_data():
    from slotpred import world as w
    from slotpred.config import Q_CANDIDATES, TEST_START
    from slotpred.simulate import simulate

    def histogram(values):
        counts = defaultdict(int)
        for v in values:
            counts[round((np.floor(v / BIN) + 0.5) * BIN, 2)] += 1
        return sorted([m, n] for m, n in counts.items())

    world, raw = simulate()
    test = world[world["date"] >= TEST_START]
    cells = {f"{d}|{v}": histogram(g["expected_minutes"]) for (d, v), g in test.groupby(["department", "visit_type"])}
    return {
        "departments": {k: list(v) for k, v in w.DEPARTMENTS.items()}, "visitTypes": list(w.VISIT_TYPES),
        "reportMin": w.REPORT_MIN, "reportCap": w.REPORT_CAP,

        "elderlyAge": w.ELDERLY_AGE, "elderlyFactor": w.ELDERLY_FACTOR, "infantAge": w.INFANT_AGE,
        "infantFactor": w.INFANT_FACTOR, "teleFactor": w.TELE_FACTOR, "clinicFactor": w.CLINIC_FACTOR,
        "dayFactor": {str(k): v for k, v in w.DAY_FACTOR.items()}, "earlyHour": w.EARLY_HOUR,
        "earlyFactor": w.EARLY_FACTOR, "lateHour": w.LATE_HOUR, "lateFactor": w.LATE_FACTOR,
        "drift": {"date": w.DRIFT_DATE, "dept": w.DRIFT_DEPT, "extra": w.DRIFT_EXTRA_MIN}, "sigma": w.NOISE_SIGMA,
        "doctors": [{"name": d.name, "dept": d.dept, "speed": d.speed, "clinics": list(d.clinics), "joined": d.joined}
                    for d in w.DOCTORS],
        "qCandidates": list(Q_CANDIDATES),
        "test": {"from": TEST_START, "visits": int(len(test)), "meanExpected": round(float(test["expected_minutes"].mean()), 2),
                 "hist": histogram(test["expected_minutes"]), "cells": cells},
        "eda": eda_data(world, raw),
        "prep": prep_data(),
        "base": baseline_data(),
        "lgbm": lgbm_data(),
        "xgb": xgb_data(),
        "final": final_data(),
    }


def xgb_data():
    """A7: LightGBM vs XGBoost on June, trained the same way, and how much they agree visit by visit."""
    import time

    from slotpred import train_lgbm, train_xgb
    from slotpred.config import Q_CANDIDATES
    from slotpred.features import make_xy
    from slotpred.metrics import evaluate
    from slotpred.prepare import load_prepared

    splits, state = load_prepared()
    Xtr, ytr = make_xy(splits["train"], state)
    Xva, yva = make_xy(splits["valid"], state)
    out = {"rows": [], "agree": {}, "trees": {}, "seconds": {}}
    for name, lib in (("LightGBM", train_lgbm), ("XGBoost", train_xgb)):
        t = time.time()
        models = lib.fit_all(Xtr, ytr, Xva, yva, save=False)
        out["seconds"][name] = round(time.time() - t, 1)
        out["trees"][name] = {str(q): int(getattr(m, "best_iteration_", None) or m.best_iteration + 1) for q, m in models.items()}
        out[name] = {str(q): m.predict(Xva) for q, m in models.items()}
        out["rows"] += [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in evaluate(yva, p, float(q), name).items()}
                        for q, p in out[name].items()]
    for q in map(str, Q_CANDIDATES):
        a, b = out["LightGBM"][q], out["XGBoost"][q]
        d = a - b
        out["agree"][q] = {"mean_abs": round(float(np.mean(np.abs(d))), 3),
                           "same_slot": round(float(np.mean(np.ceil(a / 5 - 1e-9) == np.ceil(b / 5 - 1e-9))), 4),
                           "hist": np.histogram(np.clip(d, -3, 3), bins=np.arange(-3, 3.01, 0.25))[0].tolist()}
    del out["LightGBM"], out["XGBoost"]
    return out


def final_data():
    """A8: the July-August results saved by slotpred/final.py (LightGBM rows only; XGBoost was not chosen)."""
    from slotpred.config import OUT, Q_CANDIDATES
    path = OUT / "final" / "test_results.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text(encoding="utf-8"))
    keep = ("today (fixed 15)", "B3 lookup table", "LightGBM", "LightGBM + new-doctor backoff", "Oracle (floor)")
    qs = [f"{q:.2f}" for q in Q_CANDIDATES]
    return {
        "trees": {q: r["trees"][q] for q in qs}, "test_rows": r["test_rows"], "train_rows": r["train_rows"],
        "board": [b for b in r["board"] if b["method"] in keep],
        "tradeoff": {k: v for k, v in r["tradeoff"].items() if k == "today (fixed 15)" or k in {f"Model q={q}" for q in qs}},
        "slices": {q: {grp: {lvl: {m: {"on_time": v[m]["on_time"], "bias": v[m]["bias"], "n": v[m]["n"]} for m in keep[:4] if m in v}
                             for lvl, v in levels.items()} for grp, levels in r["slices"][q].items()} for q in qs},
    }


def lgbm_data():
    """A6: the q = 0.8 training curve, one visit's prediction tree by tree, the leaderboard and importance."""
    import pandas as pd

    from slotpred.config import OUT, Q_CANDIDATES
    from slotpred.features import make_xy
    from slotpred.metrics import evaluate
    from slotpred.prepare import load_prepared
    from slotpred.train_lgbm import fit_one

    splits, state = load_prepared()
    Xtr, ytr = make_xy(splits["train"], state)
    Xva, yva = make_xy(splits["valid"], state)
    models = {q: fit_one(q, Xtr, ytr, Xva, yva) for q in Q_CANDIDATES}
    m = models[0.8]
    hist = m.evals_result_
    v = splits["valid"]
    i = int(((v["department"] == "Neurology") & (v["visit_type"] == "new")).to_numpy().argmax())
    row = Xva.iloc[[i]]
    n = len(hist["valid"]["quantile"])
    trace = [round(float(m.predict(row, num_iteration=k)[0]), 2) for k in range(1, n + 1)]
    gain = {}
    for q, mq in models.items():
        g = pd.Series(mq.booster_.feature_importance("gain"), index=mq.booster_.feature_name())
        gain[str(q)] = (g / g.sum()).round(4).to_dict()
    base = pd.read_csv(OUT / "baselines_valid.csv")
    board = base[base["method"].str.match(r"B3|Oracle")][["method", "q", "pinball", "coverage", "booked_on_time", "booked_idle_min"]]
    lg = [evaluate(yva, mq.predict(Xva), q, "LightGBM") for q, mq in models.items()]
    board = pd.concat([board, pd.DataFrame(lg)[board.columns]]).round(4)
    return {
        "best": {str(q): int(mq.best_iteration_) for q, mq in models.items()},
        "train": [round(x, 4) for x in hist["train"]["quantile"]], "valid": [round(x, 4) for x in hist["valid"]["quantile"]],
        "visit": {k: (str(val) if not isinstance(val, (int, float)) else val) for k, val in splits["valid"].iloc[i][
            ["department", "visit_type", "doctor", "clinic", "age", "prior_reports", "slot_time", "date"]].items()},
        "visitActual": round(float(yva.iloc[i]), 2), "trace": trace, "start": round(float(ytr.quantile(0.8)), 2),
        "board": board.to_dict("records"), "gain": gain,
    }


def baseline_data():
    """A5: the June leaderboard, the box-size ladder, and per-visit data for the rounding lab."""
    import pandas as pd

    from slotpred.baselines import FixedSlot, GlobalQuantile, GroupQuantile, oracle_quantile
    from slotpred.config import DATA, Q_CANDIDATES
    from slotpred.features import add_time_features
    from slotpred.metrics import evaluate, pinball_loss
    from slotpred.prepare import load_prepared
    from slotpred.world import NOISE_SIGMA

    splits, _ = load_prepared()
    train, valid = add_time_features(splits["train"]), add_time_features(splits["valid"])
    y = valid["minutes"].to_numpy()
    truth = pd.read_csv(DATA / "_hidden_truth.csv")
    expected = valid[["appointment_id"]].merge(truth, on="appointment_id", how="left")["expected_minutes"].to_numpy()
    dv = ["department", "visit_type"]

    board, b2pred = [], {}
    for q in Q_CANDIDATES:
        methods = {"B0": FixedSlot(15), "B1": GlobalQuantile(q), "B2": GroupQuantile(dv, q),
                   "B3": GroupQuantile(["doctor", "visit_type"], q, min_n=30, parent=GroupQuantile(dv, q))}
        for name, m in methods.items():
            pred = m.fit(train).predict(valid)
            board.append(evaluate(y, pred, q, name))
            if name == "B2":
                b2pred[str(q)] = m.table_.round(3).to_dict()
        board.append(evaluate(y, oracle_quantile(expected, q, NOISE_SIGMA), q, "Oracle"))

    ladder = []
    for keys in (dv, ["doctor", "visit_type"], ["doctor", "visit_type", "clinic"],
                 ["doctor", "visit_type", "clinic", "weekday"], ["doctor", "visit_type", "clinic", "weekday", "hour"]):
        row = {"keys": keys}
        for min_n in (1, 30):
            m = GroupQuantile(keys, 0.8, min_n=min_n, parent=GroupQuantile(dv, 0.8)).fit(train)
            row[f"score{min_n}"] = round(pinball_loss(y, m.predict(valid), 0.8), 3)
            row["boxes"], row["median"] = int(len(m.table_)), float(m.counts_.median())
        ladder.append(row)

    boxes = sorted(b2pred[str(Q_CANDIDATES[0])])
    box_index = {b: i for i, b in enumerate(boxes)}
    cardio = train[(train["department"] == "Cardiology") & (train["visit_type"] == "new")]["minutes"]
    return {
        "board": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in board],
        "ladder": ladder,
        "boxes": [list(b) for b in boxes],
        "b2": {q: [t[b] for b in boxes] for q, t in b2pred.items()},
        "validY": [round(float(v), 2) for v in y],
        "validBox": [box_index[(d, v)] for d, v in zip(valid["department"], valid["visit_type"])],
        "cardio": [round(float(v), 2) for v in cardio],
    }


def prep_data():
    """A4: the cleaning funnel, the splits and both leakage experiments, run on the real export."""
    import pandas as pd

    from slotpred.clean import GLITCH_CLINIC, GLITCH_FROM, GLITCH_TO, add_minutes, clean, normalise_labels
    from slotpred.config import TEST_START
    from slotpred.eda import load_raw
    from slotpred.features import fit_prep
    from slotpred.split import random_split, time_split

    raw = load_raw()
    df, funnel = clean(raw)
    # each problem counted on its own, before dedup and without "first reason wins" (the A3 view)
    m = add_minutes(raw)
    in_window = (raw["clinic"] == GLITCH_CLINIC) & raw["date"].between(GLITCH_FROM, GLITCH_TO)
    alone = {"dup": int(raw.duplicated().sum()), "missing": int(m.isna().sum()),
             "whitefield": int(in_window.sum()), "range": int((m.notna() & ~m.between(1, 120)).sum())}
    removed = dict(zip(["dup", "labels", "missing", "whitefield", "range"], funnel["removed"].iloc[:5].astype(int)))

    splits = time_split(df)
    month = df["date"].str[:7]
    which = np.select([df["date"] < "2026-06-01", df["date"] < TEST_START], ["train", "valid"], "test")
    months = sorted(month.unique())
    per_month = {s: [int(((month == mo) & (which == s)).sum()) for mo in months] for s in ("train", "valid", "test")}

    def scores(train, test, key):
        med = train.groupby(key)["minutes"].median().rename("pred")
        t = test.join(med, on=key)
        t["pred"] = t["pred"].fillna(train["minutes"].median())
        err, derm = t["minutes"] - t["pred"], t["department"] == "Dermatology"
        return {"mae": round(float(err.abs().mean()), 2), "dermMae": round(float(err[derm].abs().mean()), 2),
                "dermBias": round(float(err[derm].mean()), 2), "n": int(len(t))}

    test = splits["test"]
    fits = {"train": splits["train"], "trainValid": pd.concat([splits["train"], splits["valid"]]), "all": df}
    keys = {"dept": ["department", "visit_type"], "doctor": ["doctor", "visit_type"]}
    leak = {f: {k: scores(rows, test, key) for k, key in keys.items()} | {"ageMedian": float(rows["age"].median())}
            for f, rows in fits.items()}
    r = random_split(df)
    state = fit_prep(splits["train"])
    return {
        "raw": int(len(raw)), "clean": int(len(df)), "removed": removed, "alone": alone,
        "months": months, "perMonth": per_month, "ageFill": float(state.age_fill),
        "doctorsInTrain": len(state.categories["doctor"]),
        "nikhil": {s: int((part["doctor"] == "Nikhil Verma").sum()) for s, part in splits.items()},
        "split": {"time": scores(splits["train"], test, keys["dept"]), "random": scores(r["train"], r["test"], keys["dept"]),
                  "randomJulAug": scores(r["train"], r["test"][r["test"]["date"] >= TEST_START], keys["dept"])},
        "leak": leak,
    }



def eda_data(world, raw):
    """A3: the raw export's consult minutes, labelled with the hidden truth (genuine vs error)."""
    import pandas as pd

    r = raw.merge(world[["appointment_id", "actual_minutes"]], on="appointment_id")
    start, end = pd.to_datetime(r["consult_start"]), pd.to_datetime(r["consult_end"])
    minutes = (end - start).dt.total_seconds() / 60
    has = minutes.notna()
    r, minutes = r[has], minutes[has]
    error = (minutes - r["actual_minutes"]).abs() > 0.05
    whitefield = start[has] == pd.to_datetime(r["check_in"])

    q1, q3 = minutes.quantile([0.25, 0.75])
    logm = np.log(minutes.where(minutes > 0))
    l1, l3 = logm.quantile([0.25, 0.75])
    fences = {"raw": [q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)],
              "log": [float(np.exp(l1 - 1.5 * (l3 - l1))), float(np.exp(l3 + 1.5 * (l3 - l1)))],
              "domain": [1.0, 120.0]}
    rules = {}
    for name, (lo, hi) in fences.items():
        flag = (minutes < lo) | (minutes > hi)
        rules[name] = {"flagged": int(flag.sum()), "caught": int((flag & error).sum()), "genuine": int((flag & ~error).sum()),
                       "missed": int((~flag & error).sum()), "missedWhitefield": int((~flag & error & whitefield).sum())}

    edges = 10 ** np.arange(np.log10(0.02), np.log10(330) + 0.06, 0.06)
    kind = np.where(~error, 0, np.where(whitefield, 2, 1))          # 0 genuine, 1 impossible, 2 Whitefield glitch
    idx = np.clip(np.digitize(minutes, edges) - 1, 0, len(edges) - 2)
    hist = np.zeros((len(edges) - 1, 3), int)
    np.add.at(hist, (idx, kind), 1)

    month = pd.to_datetime(r["date"]).dt.to_period("M").astype(str)
    frame = pd.DataFrame({"month": month, "clinic": r["clinic"], "m": minutes, "wf": whitefield})
    med = lambda d: d.groupby(["month", "clinic"])["m"].median().unstack().round(2)
    allrows, cleaned = med(frame), med(frame[~frame["wf"]])
    return {
        "n": int(len(minutes)), "errors": int(error.sum()),
        "fences": {k: [round(float(a), 2), round(float(b), 2)] for k, (a, b) in fences.items()}, "rules": rules,
        "hist": [[round(float(edges[i]), 4), round(float(edges[i + 1]), 4), *map(int, hist[i])] for i in range(len(hist)) if hist[i].sum()],
        "months": list(allrows.index), "clinics": list(allrows.columns),
        "medianAll": allrows.values.tolist(), "medianClean": cleaned.values.tolist(),
    }


def fill(template, **parts):
    for key, value in parts.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def link(url, fallback):
    return url or fallback


def lesson_page(lesson, built, urls, css, common_js, data_json):
    crumbs = [f'<a href="{link(urls.get("hub"), "index.html")}">All lessons</a>']
    for other in built:
        label = f'{other["id"]} · {other["name"]}'
        if other is lesson:
            crumbs.append(f'<span aria-current="page">{html.escape(label)}</span>')
        else:
            crumbs.append(f'<a href="{link(urls.get(other["id"]), other["slug"] + ".html")}">{html.escape(label)}</a>')
    header = f"""<header>
  <div class="eyebrow">Lesson {lesson["id"]} · {html.escape(lesson["stage"])}</div>
  <h1>{html.escape(lesson["id"] + " · " + lesson["name"])}</h1>
  <p class="sub">{html.escape(lesson["summary"])} Drag any slider: the formula, the numbers and the chart change together. Written notes: <code>{lesson["note"]}</code>.</p>
  <div class="crumbs">{"".join(crumbs)}</div>
</header>"""
    body = (SRC / lesson["src"]).read_text(encoding="utf-8")
    footer = "<footer>Generated by <code>notes/interactive/build.py</code> from the project's simulator. Rerun it after changing <code>slotpred/world.py</code>.</footer>"
    return (fill(HEAD, TITLE=lesson["title"], DESC=html.escape(lesson["summary"]), CSS=css)
            + fill(LIBS, DATA=data_json if lesson.get("data") else "null", COMMON_JS=common_js)
            + f'<div class="wrap">\n{header}\n{body}\n{footer}\n</div>\n')


def hub_page(urls, css):
    rows = {"A": [], "B": []}
    pill = {"done": '<span class="pill done">Done</span>', "now": '<span class="pill now">In progress</span>',
            "planned": '<span class="pill">Planned</span>'}
    for les in LESSONS:
        status = les.get("status", "planned")
        built = "src" in les
        labs = "".join(f"<span>{html.escape(x)}</span>" for x in les.get("labs", []))
        open_link = (f'<a class="open" href="{link(urls.get(les["id"]), les.get("slug", "") + ".html")}">Open the lab →</a>'
                     if built else "")
        rows[les["id"][0]].append(f"""<div class="lessonrow{'' if built else ' planned'}">
  <div class="id">{les["id"]}</div>
  <div class="body"><div class="stage">{html.escape(les["stage"])}</div><h3>{html.escape(les["name"])}</h3>
    <p class="sub">{html.escape(les["summary"])}</p>{f'<div class="labs">{labs}</div>' if labs else ''}{open_link}</div>
  {pill[status]}
</div>""")
    body = f"""<div class="wrap">
<header>
  <div class="eyebrow">Slot prediction · interactive notes</div>
  <h1>OPD Slot Lab</h1>
  <p class="sub">One interactive page per lesson, built as the project progresses. Each lab lets you move the numbers behind an idea and watch the formula and chart respond. The written notes in <code>notes/</code> hold the full explanations.</p>
</header>
<section class="track"><h2>Track A · the production model</h2><div class="lessons">{''.join(rows['A'])}</div></section>
<section class="track"><h2>Track B · language models, for learning</h2><div class="lessons">{''.join(rows['B'])}</div></section>
<footer>Generated by <code>notes/interactive/build.py</code>.</footer>
</div>
"""
    return fill(HEAD, TITLE="OPD Slot Lab", DESC="Interactive study notes for the OPD slot-prediction project, one lab page per lesson.", CSS=css) + body


def main():
    urls = json.loads(URLS.read_text(encoding="utf-8")) if URLS.exists() else {}
    css = (SRC / "common.css").read_text(encoding="utf-8")
    common_js = (SRC / "common.js").read_text(encoding="utf-8")
    built = [les for les in LESSONS if "src" in les]
    data_json = json.dumps(slot_data(), separators=(",", ":")) if any(les.get("data") for les in built) else "null"
    for les in built:
        out = HERE / f'{les["slug"]}.html'
        out.write_text(lesson_page(les, built, urls, css, common_js, data_json), encoding="utf-8")
        print(f"wrote {out.name}")
    (HERE / "index.html").write_text(hub_page(urls, css), encoding="utf-8")
    print("wrote index.html (hub)")
    demo_page(css, common_js)


def demo_page(css, common_js):
    """The Monday deliverable: the trained model running in the page, plus the July-August results.
    Needs `python -m slotpred.final` first (outputs/final/)."""
    from slotpred.config import OUT
    final = OUT / "final"
    if not (final / "demo_model.json").exists():
        print("skipped slot-predictor.html: run python -m slotpred.final first")
        return
    data = {"model": json.loads((final / "demo_model.json").read_text(encoding="utf-8")),
            "results": json.loads((final / "test_results.json").read_text(encoding="utf-8"))}
    body = (SRC / "demo.html").read_text(encoding="utf-8")
    footer = "<footer>Built from <code>slotpred/final.py</code> and <code>notes/interactive/build.py</code>. Simulated data; a learning project.</footer>"
    page = (fill(HEAD, TITLE="OPD Slot Predictor", DESC="Book OPD consult slots from a trained LightGBM model, with results on July–August.", CSS=css)
            + fill(LIBS, DATA=json.dumps(data, separators=(",", ":")), COMMON_JS=common_js)
            + f'<div class="wrap">\n{body}\n{footer}\n</div>\n')
    (HERE / "slot-predictor.html").write_text(page, encoding="utf-8")
    print("wrote slot-predictor.html")


if __name__ == "__main__":
    main()
