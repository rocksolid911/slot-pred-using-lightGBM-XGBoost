# A2 · Simulating the hospital (lifecycle stage 1: Collect)

> **Date:** 2026-10-08 · **Notebook:** [notebooks/A02_simulator.ipynb](../notebooks/A02_simulator.ipynb) · **Code:** [slotpred/world.py](../slotpred/world.py), [slotpred/simulate.py](../slotpred/simulate.py) · **Git tag:** `lesson-A2`

## In one line
**actual minutes = hidden rule × noise**: the rule is what a model can learn; the log-normal noise is what no model can know. Because we wrote both, we know the best score any model could reach.

## Intuition

### Why simulate?
There's no real hospital data. A simulator also gives us something real projects never have: **the truth**. We can later check whether LightGBM found the rules (SHAP, A9), and we know the **oracle** score, the best any model could do.

### Rule + noise
```
actual minutes = expected_minutes(features) × noise
                 └──── the hidden rule ───┘   └ what no booking form can know ┘
```
The **noise** is everything the booking system can't see: a worried patient, a chatty patient, an unexpected finding.

### Additive or multiplicative? Ask: extra work, or a change of pace?

| Kind | Example | Formula | Meaning |
|---|---|---|---|
| **Additive** | Interpreter | m + 4 | A roughly fixed extra cost, short or long consult |
| **Multiplicative** | Slow doctor | m × 1.2 | 20% longer at everything: 10 → 12, 25 → 30 |

| Factor | Type | Size | Why |
|---|---|---|---|
| Base (department × visit type) | — | table in `world.py` | |
| Prior reports | Additive | +1.3 min each, at most 8 | Reading a report takes the same time in any consult; beyond 8 the doctor skims |
| Interpreter *(my choice)* | Additive | +4 min | Translating is a roughly fixed cost |
| Age 65+ *(my choice)* | Multiplicative | ×1.15 | Everything is slower: history, exam, explaining |
| Baby under 3 (Paediatrics) | Multiplicative | ×1.15 | Same idea as age |
| Teleconsult | Multiplicative | ×0.80 | No physical exam |
| Doctor's speed | Multiplicative | 0.85–1.35 | Personal pace applies to everything |
| Clinic | Multiplicative | 0.90–1.05 | Busy clinics run faster |
| Monday / Saturday | Multiplicative | ×0.94 / ×1.05 | Monday backlog; relaxed Saturday |
| Before 10:00 / from 16:00 | Multiplicative | ×1.05 / ×0.92 | Fresh morning; hurrying late afternoon |
| Dermatology from 1 July (**drift**) | Additive | +3 min | A fixed checklist |

**Order matters:** a slow doctor is slow at the extra work too, so the pace multiplies the reports and the interpreter time. The drift checklist is added *after* the pace, because it takes 3 minutes whoever the dermatologist is.

### Drift, built in on purpose
From **1 July** (the start of the test period): Dr Nikhil Verma joins Paediatrics (slow, never seen in training), and Dermatology adds +3 minutes. Training uses Jan–May and validation uses June, so neither change is visible until the test. That's what monitoring (A12) is meant to catch.

### What "multiplicative log-normal noise" means, word by word
- **Noise:** the random part no booking form can know.
- **Multiplicative:** the noise *scales* the expected time instead of adding to it.
  ```
  additive:        actual = expected + noise     (14 + 3)
  multiplicative:  actual = expected × noise     (14 × 1.2)
  ```
  ±3 minutes is huge for a 6-minute follow-up and tiny for a 24-minute neurology consult; ±20% is reasonable for both.
- **Log-normal:** draw $Z$ from a bell curve, then take $e^Z$.

| $Z$ drawn | noise $= e^Z$ | A 14-min visit becomes |
|---|---|---|
| −0.3 | 0.74 | 10.4 min (26% shorter) |
| 0 | 1.00 | 14.0 min |
| +0.3 | 1.35 | 18.9 min (35% longer) |
| +0.6 | 1.82 | 25.5 min |

Two consequences:
1. $e^Z$ is **always positive**, so a visit can never take negative minutes. Normal noise could produce that.
2. **Equal steps in $Z$ are unequal in minutes.** −0.3 shortens by 26%, +0.3 lengthens by 35%: short visits get squeezed, long ones stretched. That's the **right skew** from A1.

### Where σ = 0.28 comes from
It's an **assumption**, not data. It controls how much visits of the same kind vary:

| σ | 80% of visits expected to take 14 min fall between | Feels like |
|---|---|---|
| 0.10 | 12.3 and 15.8 min | Unrealistically regular, like a machine |
| **0.28** | **9.4 and 19.3 min** | Plausible: most visits close, some run long |
| 0.60 | 5.4 and 25.2 min | Very chaotic |

![Same visit type, three amounts of noise](img/a02_sigma_compare.png)

**With real data you measure σ instead of guessing it:** after fitting a model, compute $\log(\text{actual}) - \log(\text{predicted})$ for every visit; the standard deviation of those differences is σ. σ also sets the **oracle floor**: more noise means even a perfect model has a bigger error.

## Maths

### The rule
$$
\text{work} = \text{base} + 1.3\,\min(\text{reports}, 8) + 4\cdot[\text{interpreter}]
$$
$$
\text{pace} = f_{age}\, f_{infant}\, f_{tele}\, f_{doctor}\, f_{clinic}\, f_{day}\, f_{time}
$$
$$
\text{expected} = \text{work}\times\text{pace} + 3\cdot[\text{Dermatology and date} \ge \text{1 July}]
$$

### Log-normal noise
$$
Z \sim \mathcal{N}(\mu, \sigma^2), \qquad \text{noise} = e^{Z}, \qquad \text{actual} = \text{expected} \times \text{noise}
$$

| Quantity | Formula | σ = 0.28 |
|---|---|---|
| Mean of the noise | $e^{\mu + \sigma^2/2}$ | set to **1.000** |
| ⇒ centre of $Z$ | $\mu = -\sigma^2/2$ | **−0.039** |
| Median of the noise | $e^{\mu}$ | **0.962** |
| q-th quantile of the noise | $e^{\mu + z_q\,\sigma}$ | P80: $e^{0.196}$ = **1.217** |
| P80 ÷ median | $e^{z_{0.8}\,\sigma}$ | $e^{0.236}$ = **1.27** |

**Why μ = −σ²/2?** Because of the squeeze/stretch asymmetry, $e^Z$ averages slightly *above* 1 when $Z$ is centred at 0: the stretches outweigh the squeezes. Shifting the centre down by $\sigma^2/2 = 0.039$ cancels that exactly, so the noise averages 1.000 and doesn't secretly make every visit longer.

**Why is P80 about 27% above the median for every visit?** Because P80 ÷ median $= e^{\mu + z\sigma} / e^{\mu} = e^{z\sigma}$; μ and the expected minutes cancel out. The safety margin depends **only on σ**: more noise means a bigger margin.

### z-score
$$
z = \frac{\text{value} - \text{mean}}{\text{standard deviation}}
$$
How many standard deviations a value is from the mean. Example: heights average 165 cm with SD 10 cm, so 180 cm gives $z = (180-165)/10 = 1.5$.

Every normal distribution has the same shape when measured in SDs, so each $z$ marks a fixed share below it:

| $z$ | Share below it |
|---|---|
| 0 | 50% (median) |
| **0.8416** | **80%** |
| 1.2816 | 90% |
| 1.645 | 95% |

In Python: `scipy.stats.norm.ppf(0.8)` returns 0.8416.

### The P80 of the noise, step by step

![Z is a symmetric bell; e^Z converts it; the result is right-skewed](img/a02_lognormal_noise.png)

1. **Left:** $Z$ is a symmetric bell centred at μ = −0.039 with spread σ = 0.28. Its 80th percentile is 0.8416 SDs above the centre: $-0.039 + 0.8416 \times 0.28 = 0.196$.
2. **Middle:** convert with $e^Z$. Go up from 0.196 and across: $e^{0.196} = 1.217$. The upward bend of the curve is what stretches the right tail.
3. **Right:** the noise is right-skewed. The **median (0.962) sits left of the mean (1.000)**, the skew from A1.

So 80% of visits finish within 1.217 × expected. A 15-minute expected visit: P80 = 18.3 min, so a **20-minute slot**.

**Key idea:** $e^Z$ only ever increases as $Z$ increases, so it never reorders the draws, and **percentiles carry straight across**: the 80th percentile of $e^Z$ is $e^{(\text{80th percentile of } Z)}$. **The mean doesn't carry across:** $e^{(\text{mean of } Z)} = e^{-0.039} = 0.962$ is the *median* of the noise, not its mean. That's exactly why μ had to be shifted by −σ²/2.

A million random draws confirm it: mean 1.000, median 0.962, P80 1.218.

**Worked example** (the sentence from the very first message): *Siddhant (34), new appointment with Dr Deepak (General Medicine, speed 1.0), Monday 10:30 at HSR, 2 reports.*
```
work     = 12 + 1.3 × 2 + 0      = 14.6
pace     = 1.0 × 1.00 × 0.94     = 0.94      (only Monday is not neutral)
expected = 14.6 × 0.94           = 13.7 min
median   = 13.7 × 0.962          = 13.2 min
P80      = 13.7 × 1.217          = 16.7 min   ->  20-min slot
```

## Code
| Piece | Where | Written by |
|---|---|---|
| Constants: departments, doctors, factors | [world.py](../slotpred/world.py) | Claude |
| `expected_minutes(...)`, the hidden rule | [world.py](../slotpred/world.py) | me (TODO) |
| `sample_noise(rng, n, sigma)` | [world.py](../slotpred/world.py) | me (TODO) |
| Calendar, patients, timestamps, export | [simulate.py](../slotpred/simulate.py) | Claude |
| Messy-data injection (**spoiler until A3**) | `_messy_export.py` | Claude |
| Behaviour tests | [tests/test_world.py](../tests/test_world.py) | Claude |
| The oracle | [A02_simulator.ipynb](../notebooks/A02_simulator.ipynb) | me (TODO) |

## Our results
<!-- To fill in after the notebook: tests passing, number of visits, oracle vs fixed 15-min slot. -->

| | Oracle | Fixed 15-min slot |
|---|---|---|
| MAE (min) | | |
| Coverage | | |
| Average idle (min) | | |

## Gotchas
- **Percentiles survive $e^Z$; the mean doesn't.** $e^{\text{mean of } Z}$ is the median of the noise, not its mean.
- σ is an **assumption** in a simulator. With real data, **measure** it from log residuals.
- The side chat's table said "6 and 30 min" for σ = 0.60; the exact values are 5.4 and 25.2. Check quick estimates numerically.

## Self-check

<details><summary>Q1. Why multiplicative noise rather than additive noise?</summary>

A long consult can go wrong by more minutes than a short one. ±20% suits both a 6-minute follow-up and a 24-minute consult; ±3 minutes doesn't. Multiplicative log-normal noise also can't go negative and is right-skewed.
</details>

<details><summary>Q2. Why μ = −σ²/2 and not μ = 0?</summary>

With μ = 0 the noise would average e^(σ²/2) ≈ 1.04, secretly making every visit 4% longer. μ = −σ²/2 makes the mean exactly 1.
</details>

<details><summary>Q3. With σ = 0.28, what's P80 ÷ median? Does it depend on the department?</summary>

e^(0.8416 × 0.28) ≈ 1.27, and no: the expected minutes cancel. Only σ matters.
</details>

<details><summary>Q4. What z-score has 90% of a normal distribution below it?</summary>

1.2816.
</details>

<details><summary>Q5. Interpreter: additive or multiplicative, and why?</summary>

Additive. Translating adds roughly the same minutes whatever the consult's length.
</details>

<details><summary>Q6. How would you estimate σ from real data?</summary>

Fit a model, take log(actual) − log(predicted) for every visit, and compute the standard deviation.
</details>

## In my words
<!-- Siddhant: 3 lines, your own words. -->

## New terms
Simulator, hidden rule, noise, additive effect, multiplicative effect, log-normal, σ (sigma), z-score, oracle, irreducible error, drift
