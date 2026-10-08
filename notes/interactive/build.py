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
    dict(id="A3", status="now", name="Exploring the messy export (EDA)", stage="Stage 2 · Explore", summary="Distributions, missing values, odd timestamps and leakage smells."),
    dict(id="A4", name="Cleaning, features, time split", stage="Stage 3 · Prepare", summary="Fix the export, build booking-time features, split Jan–May / Jun / Jul–Aug."),
    dict(id="A5", name="Baselines", stage="Stage 4 · Baseline", summary="Fixed 15 minutes and department × visit-type medians: the bar to beat."),
    dict(id="A6", name="LightGBM quantile model", stage="Stage 5 · Train", summary="Gradient boosting with the pinball loss, one model per q."),
    dict(id="A7", name="XGBoost comparison", stage="Stage 5 · Train", summary="Same idea, different library; keep the winner."),
    dict(id="A8", name="Evaluation", stage="Stage 6 · Evaluate", summary="Slices, calibration, error analysis, and the PO's trade-off table."),
    dict(id="A9", name="Tuning, SHAP, fairness", stage="Stage 7 · Select", summary="Did the model find the hidden rule? Model card."),
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

HEAD = """<title>{{TITLE}}</title>
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

    world, _ = simulate()
    test = world[world["date"] >= TEST_START]
    cells = {f"{d}|{v}": histogram(g["expected_minutes"]) for (d, v), g in test.groupby(["department", "visit_type"])}
    return {
        "departments": {k: list(v) for k, v in w.DEPARTMENTS.items()}, "visitTypes": list(w.VISIT_TYPES),
        "reportMin": w.REPORT_MIN, "reportCap": w.REPORT_CAP, "interpreterMin": w.INTERPRETER_MIN,
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


if __name__ == "__main__":
    main()
