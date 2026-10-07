# OPD slot prediction

Predict how long an outpatient consult will take, and so which slot to book.

```
"Siddhant has a new appointment with Dr Deepak on Monday at HSR clinic"  ->  typical 14 min, safe (P80) 19 min  ->  book 20 min
```

This is a learning project with two tracks:

- **Track A, production:** a LightGBM/XGBoost quantile model. This is what gets deployed: trained, evaluated, served with FastAPI, monitored and retrained.
- **Track B, learning:** language models built up step by step, from pretraining a tiny GPT from scratch to continued pretraining, SFT and RL (GRPO) on Qwen3-0.6B. It ends with a face-off against Track A.

The lessons and revision notes are in [notes/](notes/README.md).

## Layout

| Path | Contents |
|---|---|
| `slotpred/` | Reusable code (a Python package). Notebooks and scripts import it. |
| `notebooks/` | One notebook per lesson, for exploring, plotting and running things |
| `tests/` | Checks for the parts you write yourself (`TODO(you)`) |
| `notes/` | Lesson notes, glossary, formula sheet |
| `data/`, `outputs/` | Generated files (not in git; rerun the code to recreate them) |

**Rule of thumb:** notebooks are for *looking* at things. Anything you would reuse goes into `slotpred/` and gets a test.

## Setup

**Track A** (CPU, lightweight):

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements-track-a.txt -e .
```

In VS Code, pick the kernel **"Python (slotpred Track A)"** for Track A notebooks.

**Track B** (GPU) uses the existing Unsloth environment:
`C:\Users\siddh\.unsloth\studio\unsloth_studio\Scripts\python.exe`

## Checking your work

```bash
.venv/Scripts/python.exe -m pytest
```

A failing test on a `TODO(you)` means the part isn't right yet. Passing means it is.
