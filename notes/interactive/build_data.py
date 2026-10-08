"""Embed the simulator's constants and test-period data into the interactive notes page.

    .venv/Scripts/python.exe notes/interactive/build_data.py

Rewrites the data block inside notes/interactive/index.html (window.SLOT_DATA), so the page
is one self-contained file. Rerun it whenever world.py changes, so the page always shows
the same hospital as the code.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from slotpred import world as w
from slotpred.config import Q_CANDIDATES, TEST_START
from slotpred.simulate import simulate

PAGE = Path(__file__).with_name("index.html")
BLOCK = re.compile(r"/\*SLOT_DATA\*/.*?/\*END_SLOT_DATA\*/", re.S)
BIN = 0.5


def histogram(values):
    """Compress expected minutes into [bin centre, count] pairs (only non-empty bins)."""
    counts = defaultdict(int)
    for v in values:
        counts[round((np.floor(v / BIN) + 0.5) * BIN, 2)] += 1
    return sorted([m, n] for m, n in counts.items())


def main():
    world, _ = simulate()
    test = world[world["date"] >= TEST_START]

    cells = {}
    for (dept, visit), g in test.groupby(["department", "visit_type"]):
        cells[f"{dept}|{visit}"] = histogram(g["expected_minutes"])

    data = {
        "departments": {k: list(v) for k, v in w.DEPARTMENTS.items()},
        "visitTypes": list(w.VISIT_TYPES),
        "reportMin": w.REPORT_MIN, "reportCap": w.REPORT_CAP, "interpreterMin": w.INTERPRETER_MIN,
        "elderlyAge": w.ELDERLY_AGE, "elderlyFactor": w.ELDERLY_FACTOR,
        "infantAge": w.INFANT_AGE, "infantFactor": w.INFANT_FACTOR, "teleFactor": w.TELE_FACTOR,
        "clinicFactor": w.CLINIC_FACTOR, "dayFactor": {str(k): v for k, v in w.DAY_FACTOR.items()},
        "earlyHour": w.EARLY_HOUR, "earlyFactor": w.EARLY_FACTOR,
        "lateHour": w.LATE_HOUR, "lateFactor": w.LATE_FACTOR,
        "drift": {"date": w.DRIFT_DATE, "dept": w.DRIFT_DEPT, "extra": w.DRIFT_EXTRA_MIN},
        "sigma": w.NOISE_SIGMA,
        "doctors": [{"name": d.name, "dept": d.dept, "speed": d.speed, "clinics": list(d.clinics),
                     "joined": d.joined} for d in w.DOCTORS],
        "qCandidates": list(Q_CANDIDATES),
        "test": {"from": TEST_START, "visits": int(len(test)),
                 "meanExpected": round(float(test["expected_minutes"].mean()), 2),
                 "hist": histogram(test["expected_minutes"]), "cells": cells},
    }
    block = "/*SLOT_DATA*/window.SLOT_DATA = " + json.dumps(data, separators=(",", ":")) + ";/*END_SLOT_DATA*/"
    html = PAGE.read_text(encoding="utf-8")
    if not BLOCK.search(html):
        raise SystemExit(f"no /*SLOT_DATA*/ block found in {PAGE}")
    PAGE.write_text(BLOCK.sub(lambda _: block, html), encoding="utf-8")
    print(f"updated {PAGE.name}: {len(block) / 1024:.0f} KB of data, {len(test):,} test visits, {len(cells)} cells")


if __name__ == "__main__":
    main()
