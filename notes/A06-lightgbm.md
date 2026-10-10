# A6 · LightGBM: let the computer choose the boxes (lifecycle stage 5: Train)

> **Date:** 2026-10-10 · **Notebook:** [notebooks/A06_lightgbm.ipynb](../notebooks/A06_lightgbm.ipynb) · **Code:** [mini_boost.py](../slotpred/mini_boost.py), [train_lgbm.py](../slotpred/train_lgbm.py) · **Git tag:** `lesson-A6`

## Where this fits (mind map)

```
0 Frame    → predict a SAFE slot (the q-point); q is the PO's setting (0.67 / 0.75 / 0.80)
1 Collect  → simulator made the visits
2 Explore  → found the problems
3 Prepare  → fixed them; split: train Jan–May, valid June, test Jul–Aug; built the 11 feature columns
4 Baseline → best simple method B3: 1.349 on June (q = 0.8); best possible 1.130
5 Train    → HERE: boosting, by hand and with LightGBM
```

## In one line
**Boosting = start with one number for everyone, then add many small trees; each tree finds boxes of visits that are still wrong in the same direction and nudges them a small step.**

## Intuition (one idea at a time)

### Idea 1: a tree is a box-maker that picks its own questions
In A5 **we** chose the boxes (department + visit type). A **decision tree** asks yes/no questions, like "20 questions": *is it a follow-up? which doctor? reports ≤ 3?* It tries every column and every cut point and keeps the question that lowers the score most, then repeats inside each half.

- **leaf** = the end of a branch = one box with one number.
- `min_child_samples = 30` → every box needs at least 30 training visits: **A5's backoff rule, built in**.

### Idea 2: boosting = many small trees, each fixing the last one's mistakes
1. **Start:** one number for everyone, the training P80 = **17.95** (A5's B1).
2. **Each round:** for every visit, a **push**: up by **q** if it would run late, down by **1 − q** if the doctor would sit idle.
3. A small tree groups visits that need a similar push into boxes.
4. Each box moves a **small step** (the **learning rate**, 0.05 = 5% of the way) towards the q-point of its own errors.

Late pushes (0.8) are 4 times stronger than idle pushes (0.2), so predictions settle where about 80% of visits fit: the P80.

**One June visit** (Neurology, new, Dr Meera Pillai, 2 reports, Tuesday 11:00):

```
start 17.95 → 1 tree 18.82 → 2 trees 19.61 → 3 trees 20.26 → 10 trees 23.48 → 50 trees 27.67 → 184 trees 27.55
```

### Idea 3: stop at the right time (early stopping)
More trees always improve the **training** score; eventually the trees start memorising individual training visits (**overfitting**, like tiny boxes in A5). So after each tree we check **June**; when June hasn't improved for 100 trees, we stop and keep the best number of trees. Allowed, because June is the validation split, meant for choosing settings.

## Maths (every symbol explained)

### The push (the "gradient" of the pinball score)

$$
\text{push}_i=\begin{cases}+q & y_i>p_i \text{ (late)}\\ -(1-q) & y_i<p_i \text{ (idle)}\\ 0 & y_i=p_i\end{cases}
$$

| Symbol | Meaning | Where it comes from |
|---|---|---|
| $y_i$ | how long visit $i$ really took | `minutes` column, training rows |
| $p_i$ | the model's current prediction for visit $i$ | starts at the training P80, changes every round |
| $q$ | the P value | the PO (A1), `config.Q_CANDIDATES` |

Example, q = 0.8: visits of 30 and 12 minutes, current prediction 20 → pushes **+0.8** and **−0.2**.

### One round's update

$$
p_i \leftarrow p_i + \eta \cdot v_{\text{box}(i)},\qquad v_{\text{box}} = \text{q-point of } (y_j - p_j) \text{ for the visits } j \text{ in that box}
$$

| Symbol | Meaning |
|---|---|
| $\eta$ (eta) | the learning rate: 0.1 in MiniBoost, 0.05 in LightGBM |
| $\text{box}(i)$ | the leaf that visit $i$ lands in this round |
| $v_{\text{box}}$ | how far the box would have to move so that a fraction q of its visits fit |
| $\leftarrow$ | "becomes" |

Example: a box's errors have q-point +10 minutes; with $\eta = 0.05$ every visit in it moves **+0.5** minutes this round.

## Code
| Piece | Where | Written by |
|---|---|---|
| `pinball_push`, `MiniBoost.fit` (the boosting loop) | [mini_boost.py](../slotpred/mini_boost.py) | me |
| `leaf_values`, `step`, `predict`, `to_numeric` | [mini_boost.py](../slotpred/mini_boost.py) | Claude |
| `make_model`, `fit_one` (early stopping on June) | [train_lgbm.py](../slotpred/train_lgbm.py) | me |
| `PARAMS`, `fit_all`, saving and loading models | [train_lgbm.py](../slotpred/train_lgbm.py) | Claude |
| June curves, leaderboard, one-visit trace, importance | [A06_lightgbm.ipynb](../notebooks/A06_lightgbm.ipynb) | me |

## Our results
<!-- From your notebook: MiniBoost June scores by round, the LightGBM leaderboard against A5, the training curve. -->

## Gotchas
-

## Self-check

<details><summary>Q1. <code>min_child_samples = 30</code> is the same idea as which A5 rule?</summary>

Backoff: a box (leaf) needs at least 30 training visits, so no box's number rests on a handful of visits.
</details>

<details><summary>Q2. Learning rate 0.05; a tree thinks a visit needs +10 minutes. How far does that tree move it?</summary>

0.05 × 10 = 0.5 minutes. Many small steps add up; no single tree can do much damage.
</details>

<details><summary>Q3. Why stop adding trees when June stops improving?</summary>

After that point the new trees mostly memorise training visits (overfitting): training keeps improving, June gets worse. We keep the number of trees where June, the unseen month, was best.
</details>

<details><summary>Q4. Who decides q, and who decides the learning rate?</summary>

q is the PO's business setting (A1). The learning rate, number of leaves and minimum box size are our settings, checked on June and tuned in A9.
</details>

## In my words
<!-- Written by you, not Claude: 3 lines, as if teaching a friend. -->

## New terms
decision tree, leaf, round, push (gradient), boosting, learning rate, overfitting, early stopping, feature importance (gain), hyperparameter
