# A5 · Baselines: the score to beat (lifecycle stage 4: Baseline)

> **Date:** 2026-10-10 · **Notebook:** [notebooks/A05_baselines.ipynb](../notebooks/A05_baselines.ipynb) · **Code:** [metrics.py](../slotpred/metrics.py), [baselines.py](../slotpred/baselines.py) · **Git tag:** `lesson-A5`

## Where this fits (mind map)

```
0 Frame    → what we predict: a SAFE slot length (the P-point), set by the PO's late-vs-idle ratio
1 Collect  → simulator made the visits
2 Explore  → found 8 problems
3 Prepare  → fixed them; split by date: train Jan–May, valid June, test Jul–Aug
4 Baseline → HERE: simple predictions with no ML, scored on June
5 Train    → LightGBM must beat this score (A6)
```

## In one line
**Before training anything clever, measure what simple rules already achieve**, on June, with the same score the real model will face.

## Intuition

### What a baseline is for
1. **The bar:** a score means nothing alone. "LightGBM 1.25 vs lookup table 1.40" means something.
2. **A sanity check:** if a fancy model loses to a lookup table, something is broken.
3. **The fallback:** if the model service is down (A11), the scheduler still needs a slot.

### Words

| Word | Plain meaning | Example |
|---|---|---|
| actual `y` | how long the visit really took (`minutes` column) | 25 |
| prediction `p` | the predicted P-point for that visit | 16.6 |
| `q` | the P value as a fraction: 0.67, 0.75 or 0.80. A **business setting**: from the PO's late-vs-idle ratio in A1, q = late cost ÷ (late cost + idle cost). We compute everything for all three; **the PO picks in A8** | 0.8 |
| late | minutes past the prediction: max(y − p, 0) | 25 − 20 = 5 |
| idle | minutes the doctor sat free: max(p − y, 0) | 20 − 15 = 5 |
| coverage | share of visits that fit: y ≤ p. A good P80 has about 0.80 | |
| slot | what gets booked: p rounded **up** to 5 minutes | 16.6 → 20 |
| box | one combination of columns, e.g. (Dermatology, follow-up) | |
| backoff | a box with too few visits uses the bigger box it belongs to | |

### The four baselines (each learns only from the January–May visits)

| | Method | What it does | Numbers learned |
|---|---|---|---|
| B0 | Fixed 15 | 15 minutes for everyone (today's rule) | 0 |
| B1 | One number | sort all 24,818 training visits, take the q-point | 1 |
| B2 | Lookup table | one box per department + visit type (8 × 3), take each box's q-point | 24 |
| B3 | Smaller boxes | one box per doctor + visit type; a box under 30 visits backs off to B2 | 60 |

### Why not make the boxes as small as possible?
Restaurant reviews: a rating from 4 reviews is luck, from 400 you can trust it. Very small boxes (doctor + clinic + weekday + **hour**) hold about **4 visits** each, so their P-point is mostly luck and the score gets **worse**. Backoff fixes that.

## Maths (every symbol explained)

### The score: pinball loss

$$
\text{score} = \frac{1}{n}\sum_{i=1}^{n}\Big(q\cdot\text{late}_i + (1-q)\cdot\text{idle}_i\Big)
$$

| Symbol | Meaning | Where it comes from |
|---|---|---|
| $n$ | number of visits scored | the June (valid) split: 5,515 |
| $i$ | one visit, numbered 1 to $n$ | |
| $\text{late}_i$ | $\max(y_i - p_i, 0)$ | $y_i$ from the `minutes` column, $p_i$ from the method |
| $\text{idle}_i$ | $\max(p_i - y_i, 0)$ | same |
| $q$ | 0.67, 0.75 or 0.80 | the PO (A1), via `config.Q_CANDIDATES` |
| $\sum$ | "add up over all visits" | |

**Worked example** ($q = 0.8$, $p = 20$): a 30-minute visit → late 10 → $0.8 \times 10 = 8.0$. A 12-minute visit → idle 8 → $0.2 \times 8 = 1.6$. Average of the two: $(8.0 + 1.6)/2 = 4.8$.

This is A1's cost: one late minute costs $q$, one idle minute costs $1-q$, so late is $q/(1-q)$ times worse (0.8/0.2 = 4 times for P80).

### Why each box uses its q-point
For one box, the best single number under this score is the box's q-point: sort its training visits and stop q of the way along (A1 showed why). B2 is "one best number per box, with boxes chosen by us". **LightGBM (A6) also builds boxes, but chooses them itself from all the columns.**

### How shaky is a box's P80?
Take the Cardiology new visits from train (1,302 visits, P80 = 30.2 min). Draw a small random sample, compute its P80, and repeat 2,000 times. How much do the answers jump around (their standard deviation)?

| Visits in the sample | 10 | 30 | 100 | 300 | 1,000 |
|---|---|---|---|---|---|
| P80 jumps by about (min) | 4.1 | 2.5 | 1.4 | 0.8 | 0.45 |

4 times more visits → half the jumpiness ($1/\sqrt{n}$). That's why tiny boxes are luck.

## Code
| Piece | Where | Written by |
|---|---|---|
| `pinball_loss`, `coverage`, `to_slot`, `booked_metrics` | [metrics.py](../slotpred/metrics.py) | me |
| `evaluate` (one leaderboard row) | [metrics.py](../slotpred/metrics.py) | Claude |
| `GroupQuantile.fit`, `GroupQuantile.predict` (backoff) | [baselines.py](../slotpred/baselines.py) | me |
| `FixedSlot`, `GlobalQuantile`, `oracle_quantile` | [baselines.py](../slotpred/baselines.py) | Claude |
| Leaderboard, box-size ladder, rounding table | [A05_baselines.ipynb](../notebooks/A05_baselines.ipynb) | me |

## Our results
<!-- From your notebook: the leaderboard (all three q), the box-size ladder, the rounding table. -->

## Gotchas
- Quick check: idle minutes are slot − actual (20 − 12 = 8), not the visit length. Score = 8 × 0.2 = 1.6.

## Self-check

<details><summary>Q1. q = 0.8, slot 20. Scores for a 30-minute and a 12-minute visit?</summary>

30 → 10 late × 0.8 = 8.0. 12 → 8 idle × 0.2 = 1.6. The method's score is the average over all visits.
</details>

<details><summary>Q2. Why is one number for everyone (B1) a bad plan?</summary>

Neurology new visits (P80 about 37 min) run late almost every time in a 20-minute slot; Dermatology follow-ups (about 9 min) leave the doctor idle about 11 minutes. One number can't fit both.
</details>

<details><summary>Q3. Why not trust a box with 4 visits, and what does backoff do?</summary>

With 4 visits the P80 is basically one of them: one unusual visit changes it completely. Backoff uses the bigger box it belongs to (department + visit type), which has hundreds of visits.
</details>

<details><summary>Q4. Where does q come from? Who picks it?</summary>

From the PO's late-vs-idle ratio (A1): q = late cost ÷ (late cost + idle cost). We compute all three candidates; the PO picks in A8 from booked numbers.
</details>

## In my words
<!-- Written by you, not Claude: 3 lines, as if teaching a friend. -->

## New terms
baseline, pinball loss, coverage, booked slot, box (group), backoff, oracle (floor), standard deviation of an estimate
