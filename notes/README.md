# Learning notes

Your revision material for the project. One file per lesson, all following [_TEMPLATE.md](_TEMPLATE.md).

**How to revise**
- **Quick (5 min):** reread the *In one line* and *In my words* sections across all lessons, then [FORMULAS.md](FORMULAS.md).
- **Medium (30 min):** answer one lesson's *Self-check* questions before opening the answers.
- **Deep:** check out the code exactly as it was at a lesson with `git checkout lesson-A5`, rerun the notebook, then `git checkout main`.

Also see [GLOSSARY.md](GLOSSARY.md) for every term and [FORMULAS.md](FORMULAS.md) for every formula on one page.

**Play with it:** every lesson has its own interactive lab page (sliders, live formulas, charts, heatmaps), listed on the hub [OPD Slot Lab](https://claude.ai/artifact/9KwYwYZDJCDgN3XbP8vSLf): [A1 labs](https://claude.ai/artifact/Qz3MtpX7sH92HrzuFmnrDc), [A2 labs](https://claude.ai/artifact/T2dBm3dy42pdyJWyDr14mA). Offline copies are in [interactive/](interactive/); open any `.html` in a browser. After changing `slotpred/world.py`, rebuild them with `.venv/Scripts/python.exe notes/interactive/build.py`.

**Start here:** [00-ml-workflow.md](00-ml-workflow.md), the general ML project lifecycle that every lesson below fits into.

## Track A: production model (LightGBM / XGBoost)

| # | Lifecycle stage | Lesson | Note | Status |
|---|---|---|---|---|
| A1 | 0 Framing | Target, P50 vs P80, pinball loss, metrics | [A01-framing.md](A01-framing.md) | ✅ done |
| A2 | 1 Collect | Simulating the hospital (a stand-in for an EMR export) | [A02-simulator.md](A02-simulator.md) | 🛠 in progress |
| A3 | 2 Explore | EDA: distributions, missing values, leakage smells | | |
| A4 | 3 Prepare | Cleaning, features, train / validation / test split by time | | |
| A5 | 4 Baseline | Fixed 15 min; department × visit-type median | | |
| A6 | 5 Train | LightGBM with the quantile objective | | |
| A7 | 5 Train | XGBoost comparison | | |
| A8 | 6 Evaluate | Slices, calibration, error analysis | | |
| A9 | 7 Select | Tuning, SHAP, fairness slices, model card | | |
| A10 | Cross-cutting | Experiment tracking with MLflow | | |
| A11 | 8 Deploy | FastAPI, container, shadow mode | | |
| A12 | 9 Monitor | Drift and weekly performance | | |
| A13 | 10 Retrain | Champion vs challenger, rollback | | |

## Track B: language model (learning)

| # | Lesson | Note | Status |
|---|---|---|---|
| B1 | Pretraining a tiny GPT from scratch | | |
| B2 | SFT, loss masking, why pretraining helps | | |
| B3 | Continued pretraining of Qwen (LoRA, Unsloth) | | |
| B4 | SFT on Qwen | | |
| B5 | RL with GRPO, reward design | | |
| B6 | Extractor: sentence to JSON fields for LightGBM | | |
| B7 | Face-off: LLM vs LightGBM | | |
