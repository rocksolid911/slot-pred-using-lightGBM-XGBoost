# A9 · Opening up the model (lifecycle stage 7: Select)

> **Date:** 2026-10-13 · **Notebooks:** [A09_explain.ipynb](../notebooks/A09_explain.ipynb) (part 1), [A09_fairness.ipynb](../notebooks/A09_fairness.ipynb) (part 2) · **Code:** [explain.py](../slotpred/explain.py), [fairness.py](../slotpred/fairness.py) · **Lab page:** [A9 Explain Lab](https://claude.ai/artifact/YJ73pHXqaT7RqCbgMKdxRv) (offline: [interactive/A09-explain.html](interactive/A09-explain.html)) · **Git tag:** `lesson-A9`

A9 has three parts. **This note covers parts 1 and 2**; part 3 is added when you get there. Part 2 starts at [Part 2 · Is it fair to every group?](#part-2--is-it-fair-to-every-group).

| Part | Question | Status |
|---|---|---|
| 1 | Why does the model say what it says? (SHAP) | ✅ done |
| 2 | Is it fair to every group? (on-time rate by age, video, clinic…) | **now** |
| 3 | Light tuning on June only, and a one-page model card | next |


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

SHAP values show how the model builds each prediction: it starts with a baseline and then adds or subtracts a push for each feature. Because every doctor works in exactly one department, knowing the doctor already tells the model the department, so the doctor's push can include the department's effect as well as that doctor's own pace. A feature can have a small average push but a large effect for one particular visit, like video for the example visit. Video is used only for short follow-ups, usually 6–12 minutes in person, where a 20% reduction looks almost the same as subtracting about 1.5 minutes. The model therefore could not clearly distinguish the two rules and learned a roughly fixed reduction, which works for short visits but fails on the rare long video visits. The model covered the example visit correctly because its actual duration was below the P80 prediction.

## New terms
SHAP value (push), starting point (base value), additivity, explanation vs truth

---

# Part 2 · Is it fair to every group?

## Where this fits (mind map)

```
A8     final model: 86% on time at P75 on Jul–Aug (today's 15 min: 70%)
A9 p1  ✅ why it says what it says (pushes)
A9 p2  HERE: does that 86% hold for EVERY group, or is someone quietly worse off?
```

## In one line
**An average can hide a group.** Cut the results by groups that matter, check each against the average (beyond luck) and against today, and then find *where* each gap comes from.

## Intuition (one idea at a time)

### Idea 1: the average hides groups
"86% on time" is over all 11,664 test visits. A group of babies or elderly patients could sit at 60% inside that average and nobody would notice. So we cut by groups that matter: patient age, visit type, video, clinic, department, new doctor.

### Idea 2: is a gap real, or luck?
A small group wobbles by luck alone (A5: a share from *n* visits jumps around by about $\sqrt{p(1-p)/n}$). The **luck band** is two of those wobbles: about 95% of lucky wobbles stay inside it.

*Example:* overall 86%, a group of 461 babies: band = $2\sqrt{0.86 \times 0.14 / 461}$ = 0.032, so ±3.2 points. Babies are at 79%, 7 points below: more than luck.

### Idea 3: two rulers, plus the promise
- **Against the average:** is the group below it by more than its luck band? (flag ⚑)
- **Against today:** is the group on time less often than with today's 15 minutes?
- **Against the promise:** P75 promises about 75% before rounding (more after rounding up to 5 min). A group below that is not getting what the PO signed off.

### Idea 4: where a gap comes from

| Source | How to recognise it | Fixed by |
|---|---|---|
| **Drift** (the world changed in July) | the gap is in July–August but **not in June** | retraining, monitoring (A12–A13) |
| **The booking grid** (rounding up to 5 min) | the group is covered about q **before** rounding; the gap appears **after** | nothing in the model: a booking-policy question |
| **The model itself** | the group is short **even before** rounding, **in June too** | model work, or a business rule (e.g. a higher q for that group) |

**Business setting:** how big a gap is acceptable, and whether a group gets a different q, is the PO's and the clinical lead's decision, like q itself (A1). We measure honestly; they decide.

## Maths (every symbol explained)

$$
\text{band} = 2\sqrt{\frac{p(1-p)}{n}}, \qquad \text{gap} = \text{on\_time}_{\text{group}} - p, \qquad \text{flag} = \text{gap} < -\text{band}
$$

| Symbol | Meaning | Where it comes from |
|---|---|---|
| $p$ | the on-time share of **all** test visits | `visits.on_time.mean()`: 0.857 at P75 |
| $n$ | number of visits in the group | `groupby(...).size()` |
| on_time (group) | the group's on-time share | `groupby(...).on_time.mean()` |

Worked example (65+, 1,188 visits, 81.9% on time): band = $2\sqrt{0.857 \times 0.143 / 1188}$ = 0.020; gap = 0.819 − 0.857 = −0.038; −0.038 < −0.020, so **flagged**.

## Code
| Piece | Where | Written by |
|---|---|---|
| `luck_band(p, n)`, `group_table(visits, group)` | [fairness.py](../slotpred/fairness.py) | me (TODO) |
| `booked_test(q)`, `booked_june(q)`, the group columns | [fairness.py](../slotpred/fairness.py) | Claude |
| Tests | [tests/test_fairness.py](../tests/test_fairness.py) | Claude |
| Notebook sections 1–3 | [A09_fairness.ipynb](../notebooks/A09_fairness.ipynb) | me (run + answers) |

## Our results (P75; final model on July–August, A6 model on June)

| Flagged group | Visits | On time Jul–Aug | Band | On time June | June, before rounding | Today's 15 min |
|---|---|---|---|---|---|---|
| Dermatology | 1,011 | 69.7% | ±2.2 | 93.1% | 76.0% | 80.9% |
| New doctor (Nikhil) | 532 | 69.0% | ±3.0 | – | – | 41.9% |
| Babies 0–2 | 461 | 78.7% | ±3.3 | 86.5% | 68.1% | 51.4% |
| 65+ | 1,188 | 81.9% | ±2.0 | 84.8% | 72.9% | 64.9% |
| New visits | 5,049 | 82.7% | ±1.0 | 86.4% | 74.2% | 40.4% |
| Jayanagar | 2,319 | 83.6% | ±1.5 | 85.5% | 71.4% | 61.9% |
| HSR | 2,974 | 83.4% | ±1.3 | 90.0% | 75.4% | 69.4% |
| *All visits* | *11,664* | *85.7%* | | *88.6%* | *≈ 75%* | *69.7%* |

Which source explains each row is the notebook's questions. One extra fact for question 3: babies seen by the new doctor were on time 58% of the time; babies seen by known doctors 85%, against 91% for other children.

## Gotchas
- **Small groups need the luck band.** A 5-point gap is noise for 100 visits (band ±7) and real for 5,000 (band ±1).
- **"Below today" isn't automatically harm.** Compare the slot length and the promise too.
- **One cause can show up in several cuts.** The new doctor works only in Paediatrics, at HSR and Whitefield, so his gap also appears under babies and HSR. Check the overlaps before blaming a group.

## Self-check
<details><summary>Q1. A group of 100 visits is 5 points below the average (p = 0.86). Flag it?</summary>

Band = 2√(0.86 × 0.14 / 100) = 0.069, ±6.9 points. A 5-point gap is inside it: not flagged, it could be luck.
</details>

<details><summary>Q2. A gap is in July–August but not in June. Which source is most likely?</summary>

Drift: something changed in July. Retraining on recent data (A13) and monitoring (A12) are the fixes.
</details>

<details><summary>Q3. Who decides whether a group gets a higher q?</summary>

The PO with the clinical lead: it's a business and clinical trade-off (more on time for that group vs longer slots), like q itself.
</details>

## In my words (part 2)
<!-- Written by you, after the notebook: 3 lines. -->

## New terms (part 2)
fairness slice, luck band, gap, flag, drift vs grid vs model gap, promise (q before rounding)
