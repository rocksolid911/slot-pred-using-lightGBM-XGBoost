# Learning notes

Your revision material for the project. One file per lesson, all following [_TEMPLATE.md](_TEMPLATE.md).

**How to revise**
- **Quick (5 min):** reread the *In one line* and *In my words* sections across all lessons, then [FORMULAS.md](FORMULAS.md).
- **Medium (30 min):** answer one lesson's *Self-check* questions before opening the answers.
- **Deep:** check out the code exactly as it was at a lesson with `git checkout lesson-A5`, rerun the notebook, then `git checkout main`.

Also see [GLOSSARY.md](GLOSSARY.md) for every term and [FORMULAS.md](FORMULAS.md) for every formula on one page.

## Track A: production model (LightGBM / XGBoost)

| # | Lesson | Note | Status |
|---|---|---|---|
| A1 | Framing: target, P50 vs P80, metrics | [A01-framing.md](A01-framing.md) | ✅ done |
| A2 | Data: simulating the hospital | | ⏳ next |
| A3 | Cleaning and features, leakage | | |
| A4 | Baselines | | |
| A5 | LightGBM, quantile objective | | |
| A6 | XGBoost comparison | | |
| A7 | Honest evaluation | | |
| A8 | Tuning and SHAP explainability | | |
| A9 | Experiment tracking with MLflow | | |
| A10 | Serving with FastAPI | | |
| A11 | Monitoring and drift | | |
| A12 | Retraining: champion vs challenger | | |

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
