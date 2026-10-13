# A9 · Opening up the model (lifecycle stage 7: Select)

> **Date:** 2026-10-13 · **Notebook:** [notebooks/A09_explain.ipynb](../notebooks/A09_explain.ipynb) · **Code:** [explain.py](../slotpred/explain.py) · **Lab page:** [A9 Explain Lab](https://claude.ai/artifact/YJ73pHXqaT7RqCbgMKdxRv) (offline: [interactive/A09-explain.html](interactive/A09-explain.html)) · **Git tag:** `lesson-A9`

A9 has three parts. **This note covers part 1**; parts 2 and 3 are added when you get there.

| Part | Question | Status |
|---|---|---|
| 1 | Why does the model say what it says? (SHAP) | **now** |
| 2 | Is it fair to every group? (on-time rate by age, video, clinic…) | next |
| 3 | Light tuning on June only, and a one-page model card | after that |

## Where this fits (mind map)

```
A6–A7  trained the model (LightGBM)
A8     tested it once on Jul–Aug: 86% on time at P75, vs 70% for today's 15 minutes
A9     HERE, part 1: open it up. Why does it give THIS visit THIS number?
```

## In one line
**Every prediction = a starting point + one push per column, and the pushes add up exactly.** Because we wrote the simulator, we can check whether a push matches the true rule.

## Intuition (one idea at a time)

### Idea 1: starting point plus pushes
Think of a shop bill: a base price, then one line per extra, adding up to the total. The model's "bill" for a visit:

- **Starting point:** the model's average prediction, **15.7 min** (P80), the same number for every visit.
- **Push:** how many minutes one column moves *this* visit, up (+) or down (−).
- **Total:** starting point + all pushes = the prediction, exactly (checked on all 11,664 test visits: the difference is under 0.0000000000002 min).

These pushes are called **SHAP values**. LightGBM computes them itself: `booster.predict(X, pred_contrib=True)`.

### Idea 2: one visit, read like a bill
Dr Deepak Rao, **follow-up by video**, HSR, age 46, no prior reports, Tuesday 15:45 (a July–August test visit):

```
start          15.7
visit type     −3.6   (follow-up: shorter than the average visit)
doctor         −2.8   (Dr Deepak + quick General Medicine)
video          −1.5
prior reports  −1.2   (none to read)
everything else ≈ 0
= P80           6.6 min   → booked 10   (actual: 4.2 min)
```

### Idea 3: check a push against the hidden rule
The rule (A2): video multiplies the visit's minutes by **0.80**. In person this visit would take **7.0 min** (General Medicine follow-up 7 × pace 1.00). So:

```
P80 in person   7.0 × 1.217 = 8.52
P80 by video    5.6 × 1.217 = 6.82
true push       6.82 − 8.52 = −1.70          the model found −1.53
```

1.217 is A2's P80 factor, e^(−σ²/2 + 0.8416 σ) with σ = 0.28. The model never saw the rule, only past visits, and still landed within 0.2 minutes.

### Idea 4: where it does NOT match (and why that's useful)
Over all 547 video visits in July–August:

| In-person length (rule's push) | Visits | Rule's push | Model's push |
|---|---|---|---|
| short (−1.5 to 0) | 31 | −1.40 | −1.39 |
| (−2.0 to −1.5) | 174 | −1.76 | −1.54 |
| (−2.5 to −2.0) | 163 | −2.24 | −1.63 |
| (−3.0 to −2.5) | 107 | −2.73 | −1.80 |
| long (−4.0 to −3.0) | 69 | −3.36 | −1.98 |

For short visits the model is right. For long ones it takes off only about half of what it should: it learned video as **"about 1.5–2 minutes off"**, not **"20% off"**. SHAP shows *what the model learned*, not the truth; the simulator lets us see the gap. (Why it learned it that way is notebook question 4.) The effect is on the safe side: long video follow-ups get a little too much time (idle), not too little (late).

## Maths (every symbol explained)

$$
\hat y_i = s + \sum_{j} \phi_{ij}
$$

| Symbol | Meaning | Where it comes from |
|---|---|---|
| $\hat y_i$ | the model's prediction for visit $i$ (its P80) | `booster.predict(X)` |
| $s$ | the starting point: the model's average prediction | last column of `pred_contrib` |
| $\phi_{ij}$ | the push of column $j$ for visit $i$, in minutes | the other columns of `pred_contrib` |
| $\sum_j$ | add up over all 10 columns | |

**How a push is measured:** roughly, "the prediction with this visit's value for the column" minus "the prediction averaged over the values other visits have". So a push is measured **from the average visit**, not from zero. That's why a column that is the same for almost everyone gets small pushes for most visits.

**Video's true push on the q-point** (what you code in `video_push_by_rule`):

$$
\text{push} = m \times (0.80 - 1) \times e^{-\sigma^2/2 + z_q \sigma}
$$

| Symbol | Meaning | Value |
|---|---|---|
| $m$ | the visit's expected minutes in person | from the answer key (`in_person_minutes`) |
| 0.80 | video's factor | `TELE_FACTOR` in `world.py` |
| $\sigma$ | the noise's spread (A2) | 0.28 |
| $z_q$ | the z-score for q | `norm.ppf(q)`: 0.8416 for q = 0.8 |

Worked example: $7.0 \times (-0.20) \times 1.217 = -1.70$.

## Code
| Piece | Where | Written by |
|---|---|---|
| `pushes(booster, X)` | [explain.py](../slotpred/explain.py) | me (TODO) |
| `video_push_by_rule(in_person_minutes, q)` | [explain.py](../slotpred/explain.py) | me (TODO) |
| `final_model`, `test_table`, `average_push` | [explain.py](../slotpred/explain.py) | Claude |
| Tests | [tests/test_explain.py](../tests/test_explain.py) | Claude |
| Notebook sections 1–4 | [A09_explain.ipynb](../notebooks/A09_explain.ipynb) | me (run + answers) |

## Our results (final P80 model, July–August)

Average size of each column's push (minutes):

visit type 4.61 · doctor 3.08 · prior reports 1.19 · age 0.42 · clinic 0.42 · hour 0.39 · weekday 0.29 · video 0.17 · department 0.12 · age unknown 0.00

## Gotchas
- **A push is about this model, not the world.** It explains what the model learned. Only because we have the simulator's rule can we say where that differs from the truth.
- **A push is measured from the average visit**, not from zero.
- **Small on average ≠ unimportant.** A rare column (video, about 5% of visits) has a small average push but can be big for the visits that have it.
- **Two columns that tell the same story share the credit unevenly.** The trees may use one and leave the other near zero.

## Self-check
<details><summary>Q1. What do the pushes of one visit add up to?</summary>

The prediction minus the starting point. Starting point + all pushes = the prediction, exactly.
</details>

<details><summary>Q2. The video push for a 14-minute in-person visit at P80, by the rule?</summary>

14 × (−0.20) × 1.217 = −3.41 minutes, twice the 7-minute visit's −1.70.
</details>

<details><summary>Q3. Does a big push mean that column causes longer visits?</summary>

Not necessarily. It means the model uses that column to predict longer visits. Here we can check against the rule; in a real hospital we couldn't.
</details>

## In my words
<!-- Written by you, after the notebook: 3 lines, as if teaching a friend. -->

## New terms
SHAP value (push), starting point (base value), additivity, explanation vs truth
