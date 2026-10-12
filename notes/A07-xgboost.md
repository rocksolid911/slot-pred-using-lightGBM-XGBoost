# A7 · XGBoost: a second opinion, compared fairly (lifecycle stage 5: Train)

> **Date:** 2026-10-11 · **Notebook:** [notebooks/A07_xgboost.ipynb](../notebooks/A07_xgboost.ipynb) · **Code:** [train_xgb.py](../slotpred/train_xgb.py) (written by Claude, deadline) · **Lab page:** [A7 XGBoost Lab](https://claude.ai/artifact/TaYVHKoQexdPxaqWcjhjf7) · **Git tag:** `lesson-A7`

## Where this fits

```
4 Baseline → B3: 1.349 on June (q = 0.8)
5 Train    → A6 LightGBM 1.148 · A7 XGBoost 1.163 (same data, same rules)
6 Evaluate → A8: the chosen model meets the locked test months once
```

## In one line
**Train a second model the same way; if it lands close by, trust the result more; take the better one forward.**

## Intuition
- **Why a second model:** one model's score can be luck of its settings. A different library landing close by is a second opinion.
- **Fair comparison = same everything:** same training rows (Jan–May), same 11 columns, same q values, same early stopping on June, same score. Only the library changes.
- **The one real difference:** LightGBM grows each tree **leaf by leaf** (always split the single box that helps most, capped at 31 boxes). XGBoost grows **level by level** (split every box at one depth, then the next, capped at depth 6 = up to 64 boxes).
- **Settings mapping:** `quantile_alpha` (XGBoost) = `alpha` (LightGBM) = q, the PO's setting. `min_child_weight = 30` ≈ `min_child_samples = 30` (for this objective each visit counts as weight 1), A5's backoff idea again.

## Our results (June)

| q | B3 | LightGBM | XGBoost | Best possible |
|---|---|---|---|---|
| 0.67 | 1.628 | **1.407** | 1.416 | 1.381 |
| 0.75 | 1.489 | **1.279** | 1.289 | 1.253 |
| 0.80 | 1.349 | **1.148** | 1.163 | 1.130 |

| | LightGBM | XGBoost |
|---|---|---|
| Trees kept (q = 0.67 / 0.75 / 0.80) | 140 / 137 / 184 | 537 / 593 / 477 |
| Time for all three q | 0.9 s | 4.9 s |
| Coverage at q = 0.8 | 0.791 | 0.794 |

- The two models' P80s differ by only **0.36–0.44 minutes** per visit on average; the **booked slots are identical for 91–93%** of June visits. Two libraries found the same pattern.
- **Decision: LightGBM goes forward to A8**: lower score at every q, faster, fewer trees.

## Gotchas
- XGBoost's `best_iteration` counts from 0 (add 1 for the number of trees); LightGBM's `best_iteration_` counts from 1.
- XGBoost puts `early_stopping_rounds` in the model settings, LightGBM in a callback; both watch June.

## Self-check

<details><summary>Q1. What makes a comparison between two models fair?</summary>

Same training rows, same columns, same q, same early-stopping data, same score and same evaluation month. Only the thing being compared may change.
</details>

<details><summary>Q2. Leaf-wise vs level-wise, in one sentence each?</summary>

LightGBM always splits the one box that lowers the score most; XGBoost splits every box at the current depth before going deeper.
</details>

## In my words
<!-- Written by you: which model goes forward, and why, in 2-3 lines. -->

## New terms
leaf-wise growth, level-wise growth, fair comparison, second opinion (model agreement)
