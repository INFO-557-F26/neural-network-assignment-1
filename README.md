# Assignment 1: Build the Foundation

## Overview

You are an analytics engineer at a hospital network piloting clinical machine learning.
Before anyone trains a fancy model, the **foundations** must be right:

1. Turn raw clinical notes into a usable label.
2. Build a reproducible, leakage-safe evaluation split.
3. Implement evaluation metrics from scratch — no library shortcuts.
4. Train logistic regression as your baseline.

This assignment has two parts that work together:

| Part | What it is |
|---|---|
| UI Explorer (hosted, not part of this repo) | Interactive web app for exploring the data and finding your optimal training parameters |
| `pipeline/` | Python stubs you implement in this repo to reproduce those results programmatically |

**Contents:** [Setup](#setup) · [Part 1 — UI Explorer](#part-1--ui-explorer-start-here) · [Part 2 — pipeline.py](#part-2--implement-pipelinepy) · [Running tests locally](#running-tests-locally) · [Submit your work](#submit-your-work) · [Folder structure](#folder-structure)

---

## Setup

Requires Python 3.10 or higher.

**1. Create a virtual environment**

```bash
python -m venv myenv
```

**2. Activate the environment**

On macOS / Linux:
```bash
source myenv/bin/activate
```

On Windows:
```bash
myenv\Scripts\activate
```

**3. Install the required packages**

```bash
pip install -r requirements.txt
```

---

## Part 1 — UI Explorer (start here)

The UI Explorer is an interactive browser app where you explore the dataset, understand the features, and find the best training parameters for your personal seed. It's hosted separately — nothing to install or run locally for this part.

**UI Explorer:** https://ai-sandbox-ai-healthcare.github.io/neural-network-assignments-ui-explorer/assignment-1/

### What to do in the UI Explorer

1. **Overview & Concepts tab** — Read overview & all 11 concept cards.
2. **Dataset Explorer tab** — Browse all 320 patients and understand the features and labels.
3. **Training Explorer tab** — Tune three parameters until the green **"Optimal Performance Reached!"** banner appears:
   - `learning_rate` — how large each gradient descent step is
   - `steps` — how many gradient descent steps to run
   - `val_fraction` — fraction of patients held out for validation (keep this at **0.20** — the code will reject anything else)

---

## Part 2 — Implement `pipeline.py`

### What you implement

Open `pipeline/pipeline.py` and implement the 8 stub functions:

| # | Function | What it does |
|---|---|---|
| 1 | `flag_keyword_match` | Regex keyword scan over condition text |
| 2 | `stratified_split` | Reproducible, class-balanced train/val split |
| 3 | `sigmoid` | Logistic activation |
| 4 | `logistic_regression_gradients` | Gradient of log-loss |
| 5 | `train_logistic_regression` | Full gradient descent loop |
| 6 | `confusion_counts` | TP / FP / TN / FN |
| 7 | `precision_recall_f1` | Three evaluation metrics |
| 8 | `roc_auc` | Area under the ROC curve |

Two functions (`standardize_features`, `predict_proba`) are already implemented — do not change them.

### Declare your UI Explorer parameters

Near the top of `pipeline/pipeline.py`, fill in `get_sandbox_params()` with the values you found in the UI Explorer:

```python
def get_sandbox_params() -> dict:
    return {
        "student_id":    "",      # ← your NetID, e.g. "jdoe"
        "learning_rate": 0.0,     # ← replace with value from ui exploration
        "steps":         0,       # ← replace with value from ui exploration
        "val_fraction":  0.20,    # ← keep at 0.20
    }
```

`val_fraction` must stay exactly `0.20` — `get_sandbox_params()` raises a `ValueError` if it's anything else, since the grader always uses a 20% validation split.

### Verify locally

Once you have implemented the functions and filled in `get_sandbox_params()`, run:

```bash
python pipeline/pipeline.py
```

You'll see your computed metrics for your personal seed. For example:

```
  Student : jdoe   Seed: 767
  Params  : lr=0.521, steps=330, val_fraction=0.2

  Metric    Value
  ------------------
  F1        0.745
  Accuracy  0.797
  AUC       0.840
  Loss      0.366

  Note: This local output is informational only and does not reveal hidden test code.
  The official grader will perform the final evaluation.
  Run 'pytest test_pipeline.py -v' to verify your implementation passes all tests.
  Submit your pipeline.py to GitHub for grading.
```

---

## Running tests locally

Run this to verify your implementation before pushing:

```bash
pytest pipeline/test_pipeline.py -v
```

Every test fails with `NotImplementedError` until you implement the corresponding function — that's expected for an unimplemented stub, not a bug. Run a single test while you work on one function:

```bash
pytest pipeline/test_pipeline.py::test_sigmoid -v
```

---

## Submit your work

Once your tests pass locally, commit and push your updated `pipeline.py` to GitHub:

```bash
git add pipeline/pipeline.py
git commit -m "Implement pipeline.py"
git push
```

Pushing triggers the `Autograding` GitHub Actions workflow, which runs the hidden grader. Check the **Actions** tab on your repo for the result.

---

## Folder structure

```
neural-network-assignment-1/
├── README.md                  This file
├── requirements.txt           Install with: pip install -r requirements.txt
├── .github/workflows/
│   └── autograding.yml        Runs the hidden grader on push (do not edit)
├── data/
│   └── patient_features.csv   320 synthetic patients (do not edit)
└── pipeline/
    ├── pipeline.py             The only file you edit
    └── test_pipeline.py        Unit tests (do not edit)
```
