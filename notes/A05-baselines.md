# A5 · Baselines: the score to beat (lifecycle stage 4: Baseline)

> **Date:** 2026-10-10 · **Notebook:** [notebooks/A05_baselines.ipynb](../notebooks/A05_baselines.ipynb) · **Code:** [metrics.py](../slotpred/metrics.py), [baselines.py](../slotpred/baselines.py) · **Lab page:** [A5 Baselines Lab](https://claude.ai/artifact/CgLBFQoQyi2qeuuAf75qnv) · **Git tag:** `lesson-A5`

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

All fitted on train (Jan–May, 24,818 visits), scored on valid (June, 5,515 visits).

### Leaderboard: score (pinball loss, lower is better)

| Method | q = 0.67 | q = 0.75 | q = 0.80 |
|---|---|---|---|
| B0 fixed 15 | 2.639 | 2.498 | 2.410 |
| B1 one number | 2.635 | 2.450 | 2.244 |
| B2 department + visit type | 1.700 | 1.556 | 1.404 |
| B3 doctor + visit type (backoff to B2) | **1.628** | **1.489** | **1.349** |
| Oracle (floor, uses hidden truth) | 1.381 | 1.253 | 1.130 |

- Best real method: B3. At q = 0.8 it is 0.22 above the floor; that gap is LightGBM's target in A6.
- B0's coverage is 0.688 at every q: **today's 15 minutes is secretly a P67**.
- B1's coverage is right (0.794 at q = 0.8) but its score is bad: right on average, wrong for each kind of visit.

### Box-size ladder (q = 0.8)

| Boxes by | Boxes | Median visits per box | Score, no backoff | Score, backoff (min 30) |
|---|---|---|---|---|
| department + visit type | 24 | 991 | 1.404 | 1.404 |
| doctor + visit type | 60 | 438.5 | 1.349 | 1.349 |
| + clinic | 120 | 214 | **1.335** | 1.335 |
| + weekday | 720 | 31 | 1.375 | 1.378 |
| + hour | 6,078 | 4 | **1.671** | 1.404 (= B2) |

### Rounding up to 5 minutes (B2)

| q | Coverage (raw P-point) | On time (booked slot) | Late min / visit | Idle min / visit |
|---|---|---|---|---|
| 0.67 | 0.683 | **0.803** | 0.75 | 4.29 |
| 0.75 | 0.749 | 0.866 | 0.51 | 5.59 |
| 0.80 | 0.794 | 0.885 | 0.39 | 6.13 |

Rounding adds about 9–12 points of on-time share. If the PO wants about 80% on time *as booked*, q = 0.67 already gets there. The PO decides in A8.

## Gotchas
- Quick check: idle minutes are slot − actual (20 − 12 = 8), not the visit length. Score = 8 × 0.2 = 1.6.
- `y - pred` fails on plain Python lists; convert with `np.asarray(..., float)` first.
- **Indentation decides how often a line runs.** The oracle row sat inside the per-method loop, so it was added 4 times per q.
- **Don't reuse a results list for a different table.** Appending the ladder rows to the leaderboard's `rows` put NaN rows into `baselines_valid.csv`, the file A6 reads.
- Backoff protects against tiny boxes but doesn't guarantee a gain: at the weekday level it was slightly worse (1.378 vs 1.375).

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
baseline is required for :
- to compare the models score to something for better idea how model works and also to check if pipeline works fine its not when a nice model was beaten by baseline also if the model fails for some reason we can use this as a fallback
### Why late × q + idle × (1 − q), and where q comes from

- This is the pinball loss. A late minute (slot too short) and an idle minute (slot too long) don't cost the same, and q sets the exchange rate.
-At q = 0.8, a late minute costs 0.8 and an idle minute 0.2, so lateness is four times as costly. The best single number under this score is the 80th percentile of the durations, a P80.
-  slot 20, a 30-min visit is 10 late, so 10 × 0.8 = 8.0. A 12-min visit leaves 8 idle, so 8 × 0.2 = 1.6. The average is 4.8.
- q is a business setting, not something the model chooses. It comes from the PO (A1), who picks 0.67, 0.75 or 0.80 depending on how much a patient waiting counts against an empty slot. All three are computed, and the PO picks in A8.

### why not make the boxes samll as possible
- smaller boxes are more specific, but each holds fewer visits, so its P-point is estimated from less data and becomes noisy.
- In ladder the loss improved until about 214 visits per box (1.335), then worsened: 1.375 at 31 visits per box and 1.671 at 4 visits per box.
- Backoff (min_n=30) says that if a box has fewer than 30 visits, use the parent's number (B2) instead. At the finest level this brought the loss from 1.671 back to 1.404, which is B2's score.

### what LightGBM does diffrently
- B2 uses boxes that we choose by hand (department × visit type) and gives one fixed number per box.
- LightGBM also builds boxes, but it chooses them itself from all the columns. It splits on whichever features and thresholds reduce the loss most, such as doctor, hour, age or weekday.
- It can therefore find useful boxes that a person would not think of, while using the same pinball loss for the target quantile q.
## New terms
baseline, pinball loss, coverage, booked slot, box (group), backoff, oracle (floor), standard deviation of an estimate
