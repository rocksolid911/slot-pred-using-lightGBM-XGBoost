"""The whole A4 pipeline in one command: raw export -> clean -> split by time -> fit prep on train -> save.

    .venv/Scripts/python.exe -m slotpred.prepare

Writes data/prepared/{train,valid,test}.parquet (the clean visits; features are rebuilt from them
with make_features) and data/prepared/prep_state.json (what fit_prep learned from train).
"""
import json
from dataclasses import asdict

import pandas as pd

from .clean import clean
from .config import DATA
from .eda import load_raw
from .features import PrepState, fit_prep
from .split import describe, time_split

PREPARED = DATA / "prepared"


def main():
    df, funnel = clean(load_raw())
    print(funnel.to_string(index=False), "\n")
    splits = time_split(df)
    print(describe(splits), "\n")
    state = fit_prep(splits["train"])
    PREPARED.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(PREPARED / f"{name}.parquet", index=False)
    (PREPARED / "prep_state.json").write_text(json.dumps(asdict(state), indent=1), encoding="utf-8")
    print(f"age_fill = {state.age_fill}; levels: { {k: len(v) for k, v in state.categories.items()} }")
    print(f"wrote {PREPARED}")
    return splits, state


def load_prepared():
    """(splits, state) as saved by main(). Later lessons start here."""
    splits = {n: pd.read_parquet(PREPARED / f"{n}.parquet") for n in ("train", "valid", "test")}
    state = PrepState(**json.loads((PREPARED / "prep_state.json").read_text(encoding="utf-8")))
    return splits, state


if __name__ == "__main__":
    main()
