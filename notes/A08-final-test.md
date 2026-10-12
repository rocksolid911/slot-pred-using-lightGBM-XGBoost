# A8 · The final test: one honest look at July–August (lifecycle stage 6: Evaluate)

> **Date:** 2026-10-11 · **Code:** [final.py](../slotpred/final.py) (written by Claude, deadline) · **Results page:** [OPD Slot Predictor](https://claude.ai/artifact/JHXNTJyURvBknp3ZZVPE8Q) · **Lab page:** [A8 Final Test Lab](https://claude.ai/artifact/L2zqZxrAXMbifjV7qUQPLQ) · **Git tag:** `lesson-A8`

## Where this fits

```
3 Prepare  → test months (Jul–Aug) locked away
4 Baseline → simple rules scored on June
5 Train    → LightGBM (A6) beat XGBoost (A7) on June
6 Evaluate → HERE: the chosen model meets the test months ONCE; the number we report
```

## In one line
**Choose everything on June; then refit on January–June and score once on July–August. That one score is the honest answer to "how good is it?".**

## Intuition (plain words)

### Why only once?
If you look at the test months, change something, and look again, the test months slowly become part of your choices, and the score becomes optimistic, like a student who has seen the exam. So all choices (model, settings, number of trees) were made on June; July–August was opened only now.

### Why refit on January–June first?
June was for **choosing**. Once the choices are fixed, June's 5,418 visits can also **teach**: more data, and the most recent. We keep the number of trees early stopping chose on June (122 / 107 / 126) and train on all 30,181 January–June visits. Nothing about July–August is used.

### The trade-off table: what the PO chooses from
Every visit's slot = the model's prediction rounded up to 5 minutes.

| Booking rule | On time | Late min / visit | Idle min / visit | Average slot | Patients per 4 h |
|---|---|---|---|---|---|
| Today: 15 min for everyone | 70% | 2.17 | 3.85 | 15.0 | 16.0 |
| Model P67 | 81% | 0.69 | 3.91 | 16.5 | 14.5 |
| Model P75 | 86% | 0.48 | 4.74 | 17.6 | 13.7 |
| Model P80 | 88% | 0.40 | 5.27 | 18.2 | 13.2 |

| Column | Where it comes from |
|---|---|
| On time | share of the 11,664 test visits with actual minutes ≤ booked slot |
| Late min / visit | average of max(actual − slot, 0) over all test visits |
| Idle min / visit | average of max(slot − actual, 0) |
| Patients per 4 h | 240 minutes ÷ average slot |

**Plain reading:** the model gets many more patients seen on time and cuts the next patient's waiting by about 4×. The price is longer slots, so about 1.5–3 fewer patients per 4-hour session. Which row to use is the PO's decision (q is a business setting); P75 is only our provisional default.

### Score against the alternatives (pinball, lower is better)

| q | Today (15) | Lookup table (B3) | LightGBM alone | **Model** (LightGBM + new-doctor fallback) | Best possible |
|---|---|---|---|---|---|
| 0.67 | 2.722 | 1.657 | 1.567 | **1.524** | 1.413 |
| 0.75 | 2.587 | 1.513 | 1.455 | **1.395** | 1.284 |
| 0.80 | 2.503 | 1.373 | 1.351 | **1.268** | 1.160 |

At P75 the model closes 91% of the gap between today's rule and the best possible score.

**A live example of "only once".** The test run also scored XGBoost, and on July–August XGBoost *alone* beats LightGBM *alone* (1.243 vs 1.351 at P80). It's tempting to switch now, but that would be choosing on the test months, exactly what 8.1 in the lab warns about. On June, where we choose, LightGBM was better. And with the new-doctor fallback both models land at the same score (1.268 vs 1.270), so XGBoost's lead came almost entirely from the one new doctor (point 2 below).

### What the test months revealed (things June could not show)
1. **Dermatology changed on 1 July** (+3 minutes per visit, A2's drift). The model learned January–June, so it books Dermatology too short: about 64–72% on time, while every other department is at 77–92%. Fix: retrain monthly and watch for drift (A12, A13).
2. **A new doctor** (Nikhil Verma, joined July). The model leans mostly on the `doctor` column (41% of gain). For an unknown doctor, LightGBM alone guessed blindly: only 27–36% on time. **Fix applied: A5's backoff.** A doctor not in training gets the department + visit-type lookup instead → 54–69% on time. He is also slower than his department, which only his own data can teach, after retraining. (XGBoost alone got him 86–93% on time, but by luck: its default branch for an unknown value happens to lead to long slots, which suits this one slow doctor. A fast new doctor would get slots far too long.)
3. **Coverage is below q on the test months** (0.755 for q = 0.8, vs 0.795 on June), because of 1 and 2. This is what drift looks like in numbers.

## The demo
[OPD Slot Predictor](https://claude.ai/artifact/JHXNTJyURvBknp3ZZVPE8Q) runs the trained trees inside the page: each tree is a chain of yes/no questions, and the page adds up the leaf values exactly as LightGBM does (checked: differences below 0.0000000001 minutes; `tests/test_final.py`). Example: Dr Deepak Rao, new visit, Monday 10:30 at HSR, age 34, 2 reports → P67 16.1, P75 16.6, P80 17.0 minutes → **book 20** (the P75 model alone says 16.5, a little below the P70 model's 16.6; keeping the levels in order lifts it to 16.6).

## Code
| Piece | Where | Written by |
|---|---|---|
| `choose_trees`, `refit`, test scoring, slices, trade-off table | [final.py](../slotpred/final.py) | Claude (deadline) |
| `export_trees` (model → JSON for the page) | [final.py](../slotpred/final.py) | Claude |
| Demo and results page | [src/demo.html](interactive/src/demo.html) | Claude |

Rerun everything: `.venv/Scripts/python.exe -m slotpred.final`, then `.venv/Scripts/python.exe notes/interactive/build.py`.

## Self-check

<details><summary>Q1. Why is the test split used only once?</summary>

Every look that leads to a change leaks the test months into our choices, making the score optimistic. One look = an honest estimate of the future.
</details>

<details><summary>Q2. Is refitting on January–June before the test "cheating"?</summary>

No. All choices were made on June; refitting only uses June's rows as extra training data. No July–August information is used.
</details>

<details><summary>Q3. Why did the new doctor need a fallback?</summary>

The model relies heavily on the doctor column; for a doctor it never saw, its trees take an arbitrary path. The department + visit-type lookup (A5's backoff) is a much better guess until his own visits are in the training data.
</details>

## In my words
The model improves on-time bookings to 86% at P75,
 but sees fewer patients per session; testing also revealed data drift and new-doctor risks.

## New terms
test set (used once), refit on train + valid, trade-off table, cold start fallback, drift (seen in numbers)
