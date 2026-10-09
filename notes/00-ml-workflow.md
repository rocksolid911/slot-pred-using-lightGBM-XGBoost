# 00 · The ML project workflow (for any model)

> A general reference, not specific to slot prediction. Reread it at the start of any new ML project.
> It follows the common industry lifecycle: CRISP-DM, Google's *Rules of ML*, and standard MLOps practice.

## Two different things called "training"

| | **ML lifecycle** (this page) | **Training loop** |
|---|---|---|
| Scope | The whole project, from the business question to retiring the model | One step inside stage 5 |
| Time | Weeks to years | Seconds to days |
| Steps | Frame → data → explore → prepare → baseline → train → evaluate → deploy → monitor → retrain | Forward pass → loss → backward pass (gradients) → update weights → repeat |

People say "training a model" for both. Most project failures happen **outside** the training loop: a wrong target, leakage, a bad split, no monitoring.

## The lifecycle at a glance

```
 0 Frame ──► 1 Collect ──► 2 Explore ──► 3 Prepare ──► 4 Baseline ──► 5 Train ──► 6 Evaluate
   ▲                         ▲              ▲                           ▲            │
   │                         └──────────────┴──── error analysis ───────┴────────────┤  (the inner loop:
   │                                                                                 │   most time is spent here)
   │                                                                                 ▼
   └──── problem changed ◄── 10 Retrain ◄── 9 Monitor ◄── 8 Deploy ◄── 7 Select & explain
```

It is **a loop, not a line**. Evaluation sends you back to explore, prepare or train many times before anything is deployed. Monitoring later sends you back to retrain, and sometimes all the way back to framing.

---

## 0 · Framing (problem definition)

**Purpose:** turn a business wish ("schedules run late") into a precise ML task, *before* touching data. This is always the first step, because every later step depends on it.

**Questions to answer**
- What **decision** will the model improve, and who acts on its output?
- What exactly is the **target** (the label), and how is it measured?
- **When** is the prediction made, and which inputs exist at that moment? (This rule prevents leakage.)
- What **type** of ML task is it: regression, classification, ranking, quantile, generation, extraction…?
- What does **success** mean? Give a business metric (waiting time, revenue) **and** an ML metric (MAE, F1).
- What's the **baseline** today, and does this even need ML? (A rule may be enough.)
- Constraints: latency, cost, explainability, privacy, regulation, fairness.

**Output:** a one-page spec covering the target, inputs, output format, metrics, success threshold and constraints.

**Common mistakes:** optimising a metric nobody cares about; a vague target; using inputs that won't exist at prediction time.

## 1 · Data collection and access

**Purpose:** get data that represents the situations the model will face.

**Questions:** Where does the data live? Who owns it? Is there enough? Are the labels trustworthy? What are the privacy and consent rules (for example, de-identifying patient data)? Does the data cover all the cases (departments, seasons, rare events)?

**Output:** raw data plus a **data dictionary** (what each column means and how it is recorded).

**Common mistakes:** data that only covers easy cases; labels defined differently across sources; forgetting legal approval.

## 2 · Exploratory data analysis (EDA)

**Purpose:** *understand* the data before changing it. Look at distributions, missing values, outliers, relationships with the target, and changes over time.

**Questions:** What does the target's distribution look like (skewed? multi-modal?). Which features look predictive? Are there impossible values? Does anything look suspiciously **too** predictive (a sign of leakage)?

**Output:** plots, a list of data problems, hypotheses about features.

**Common mistakes:** skipping it and finding problems only after training; exploring the **test** set (this leaks your judgement into the model).

**Full step-by-step flow for any dataset:** [00-eda-playbook.md](00-eda-playbook.md).

## 3 · Data preparation

**Purpose:** turn raw data into model-ready data.

- **Cleaning:** remove impossible values and duplicates, fix inconsistent categories, handle missing values.
- **Features:** encode categories, create useful derived inputs. For LLMs this step is **tokenisation** and formatting text into prompts.
- **Splitting** into three sets:

| Set | Used for | Rule |
|---|---|---|
| **Train** | fitting the model | — |
| **Validation** | choosing features, hyperparameters, early stopping | can be used many times |
| **Test** | the final honest score | **touch once, at the end** |

Split **by time** when the model predicts the future (train on the past, test on later data). Split **by group** (patient, user) when the same entity must not appear in both train and test.

**Output:** a reproducible pipeline (code, not manual edits) and the three splits.

**Common mistakes:** fitting cleaning or scaling steps on all the data before splitting (leakage); random splits on time-ordered data; repeatedly checking the test set until it looks good.

## 4 · Baselines

**Purpose:** set the bar. Without a baseline, "MAE 3.2" means nothing.

**Typical baselines:** today's rule (for example, a fixed 15-minute slot), a group average or median, a simple linear model.

**Output:** baseline scores on the validation set.

**Common mistakes:** skipping it, then deploying a complex model that is barely better than a median.

## 5 · Modelling and training

**Purpose:** pick a model family and a **loss function** that matches the goal, then train.

**Decisions:**
- **Model family:** gradient-boosted trees for tables; neural networks and transformers for text, images and audio.
- **Loss:** it decides *what* the model learns (mean, median, quantile, class probability).
- **Training setup:** learning rate, number of iterations, regularisation, early stopping on the validation set.

**For language models, this stage has sub-stages:**

| Stage | Data | What the model learns |
|---|---|---|
| Pretraining | Huge amounts of raw text, no labels | Language and world knowledge (next-token prediction) |
| Continued pretraining | Domain text | The vocabulary and patterns of the domain |
| SFT (supervised fine-tuning) | Input → output examples | The task and the answer format |
| Preference learning / RL (DPO, GRPO…) | Rewards or preference pairs | What "good" means, such as the business objective |

**Output:** trained model files and the logged settings that produced them.

**Common mistakes:** a loss that doesn't match the business goal; not fixing random seeds; tuning on the test set.

## 6 · Evaluation and error analysis

**Purpose:** find out honestly how good the model is and **where it fails**.

- Compare against the baseline on the **validation** set; use **test** once, at the end.
- **Slices:** break results down by subgroup (department, doctor, age group, new vs returning). A model can be good overall and bad for one group.
- **Calibration:** do predicted probabilities or quantiles mean what they claim? (Is P80 covering 80%?)
- **Error analysis:** read the worst errors one by one, find the patterns, then go back to stages 2–5.

**Output:** a metrics table against the baseline, slice breakdowns, a list of known failure modes.

**Common mistakes:** reporting only one overall number; never reading individual errors.

## 7 · Model selection, explainability and fairness

**Purpose:** choose the model to ship, and make sure you can explain and defend it.

- **Hyperparameter tuning** on the validation set.
- **Explainability:** feature importance, SHAP values. Does the model use sensible signals?
- **Fairness:** are errors much worse for some group (children, elderly patients, people who need an interpreter)?
- **Model card:** a short document covering what the model is for, its data, metrics, limits and known failure modes.

**Output:** a chosen model version, a model card, sign-off from the domain experts.

## 8 · Deployment

**Purpose:** put the model where decisions happen, safely.

- **Packaging:** an API (FastAPI) or a batch job, inside a container, with input validation.
- **Safe rollout:**
  1. **Shadow mode:** the model predicts, but nobody uses the prediction yet. Compare it with what actually happens.
  2. **Pilot:** one clinic, team or region.
  3. **A/B test:** measure the **business** metric, not just the ML metric.
  4. Full rollout.
- **Fallback:** what happens if the model is down? (Use the old rule.)

**Common mistakes:** offline metrics improve but the business metric doesn't; no way to switch the model off.

## 9 · Monitoring

**Purpose:** models degrade silently. Catch it.

| Monitor | Example |
|---|---|
| **Data drift**: inputs change | a new doctor, a new visit type, a new clinic |
| **Concept drift**: the input → output relationship changes | a department adds a protocol, so the same visit now takes longer |
| **Performance** (once the true labels arrive) | weekly MAE and coverage, by slice |
| **Business KPIs** | patient waiting time, doctor overtime |
| **System health** | latency, errors, uptime |

**Output:** dashboards and alerts with thresholds.

## 10 · Retraining and maintenance

**Purpose:** keep the model current.

- Retrain on a schedule (monthly) or when a drift alert fires.
- **Champion vs challenger:** a new model only replaces the current one if it wins on the same evaluation.
- **Versioning and rollback:** keep old versions so you can go back.
- **Retirement:** know when to switch a model off.

---

## Cross-cutting practices (in every stage)

| Practice | Why it matters |
|---|---|
| **Reproducibility**: code in git, fixed seeds, versioned data | You can recreate any result |
| **Experiment tracking** (MLflow and similar) | You know which settings produced which score |
| **Tests** for data and code | Catch broken pipelines before they produce a bad model |
| **Documentation**: spec, data dictionary, model card | Others (and future you) can trust and maintain it |
| **Domain experts in the loop** | Doctors and schedulers catch nonsense that metrics miss |
| **Privacy, ethics, governance** | Especially for health data: consent, de-identification, access control |

## How this project maps onto the lifecycle

| Stage | Track A (production, LightGBM) | Track B (learning, LLM) |
|---|---|---|
| 0 Framing | A1 | reuses A1 |
| 1 Collect | A2: **simulated** (a teaching substitute; real projects pull from the EMR) | A2 data rendered as text |
| 2 Explore | A3 | — |
| 3 Prepare | A4: cleaning, features, train/validation/test split by time | B1 tokeniser, B4 prompt format |
| 4 Baseline | A5 | A5 baselines reused |
| 5 Train | A6 LightGBM, A7 XGBoost | B1 pretraining, B2 SFT (tiny GPT), B3 continued pretraining, B4 SFT, B5 GRPO, B6 extractor |
| 6 Evaluate | A8: slices, calibration, error analysis | B7 face-off |
| 7 Select & explain | A9: tuning, SHAP, fairness slices, model card | — |
| (cross-cutting) tracking | A10: MLflow | — |
| 8 Deploy | A11: FastAPI, container, shadow mode | — |
| 9 Monitor | A12: drift, weekly performance | — |
| 10 Retrain | A13: champion vs challenger, rollback | — |
