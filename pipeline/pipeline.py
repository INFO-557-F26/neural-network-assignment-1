"""
pipeline.py — Assignment 1: Build the Foundation

You are building the data pipeline for a clinical machine learning task:
predicting which patients are likely to have chronic pain, using a synthetic
dataset of 320 patient records.

Your job is to implement 8 functions — no external ML libraries, just NumPy.
Two helper functions (standardize_features, predict_proba) are already
implemented for you; do not change them.

─────────────────────────────────────────────────────────────────────────────
FUNCTIONS YOU MUST IMPLEMENT
─────────────────────────────────────────────────────────────────────────────

  1. flag_keyword_match       Turn raw condition text into a binary pain label
                              by scanning for pain-related keywords.

  2. stratified_split         Split the dataset into train and validation sets
                              while preserving the class ratio in both halves.

  3. sigmoid                  The logistic activation: maps any real number to
                              a probability in (0, 1).

  4. logistic_regression_     Compute the gradient of the log-loss so the
     gradients                training loop knows which direction to move.

  5. train_logistic_          Run gradient descent for a fixed number of steps,
     regression               returning the trained weights and bias.

  6. confusion_counts         Count TP, FP, TN, FN between predicted and true
                              binary labels.

  7. precision_recall_f1      Compute precision, recall, and F1 from the
                              confusion counts.

  8. roc_auc                  Compute the AUC using pairwise comparison of
                              every (positive, negative) score pair.

─────────────────────────────────────────────────────────────────────────────
WORKFLOW
─────────────────────────────────────────────────────────────────────────────

  1. Complete the assignment-1 ui exploration to find your optimal hyperparameters.
  2. Fill in get_sandbox_params() with your NetID and hyperparameters from the UI.
  3. Implement all 8 functions below.
  4. Run:  'python pipeline.py'   to verify your results match the oracle.
  5. Run:  'pytest test_pipeline.py -v'   to verify your implementation passes all tests.
  6. Commit pipeline.py and your submission file, then push.

Do not modify standardize_features, predict_proba, or anything below the
'Local verification' section.
"""

import hashlib
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# ======================================================================================
# Assignment-1 UI parameters  ← fill in after completing the assignment-1 UI exploration
# ======================================================================================

def get_sandbox_params() -> dict:
    """Return your student ID and the hyperparameters you discovered in the
    assignment-1 UI exploration that reached optimal performance for your personal seed.

    After the Training Explorer shows the green 'Optimal Performance Reached!' banner, 
    copy those three values here alongside your NetID.

    Your student_id is used to derive the same seed.

    :return: dict with keys 'student_id' (str), 'learning_rate' (float),
             'steps' (int), 'val_fraction' (float).
    """
    params = {
        "student_id":    "",      # ← your NetID, e.g. "jdoe"
        "learning_rate": 0.0,     # ← replace with value from ui exploration
        "steps":         0,       # ← replace with value from ui exploration
        "val_fraction":  0.20,    # ← keep at 0.20
    }

    if params["val_fraction"] != 0.20:
        raise ValueError(
            "val_fraction must be exactly 0.20 -- do not change this value, "
            "the oracle always uses a 20% validation split."
        )
    
    return params


# ---------------------------------------------------------------------
# Do not change this list. This is not perfectly comprehensive, but
# it is a reasonable set of keywords to flag patients with chronic pain.
# ---------------------------------------------------------------------

PAIN_KEYWORDS = [
    "chronic", "pain", "arthritis", "osteoarthritis", "rheumatoid",
    "fibromyalgia", "migraine", "neuropathy", "neuralgia",
    "sciatica", "back pain", "neck pain", "spinal", "fracture",
    "injury", "burn", "wound", "trauma", "sprain", "strain",
    "tendon", "ligament", "joint", "osteoporosis", "gout",
    "lupus", "paralysis", "amputation", "surgery", "postoperative", "whiplash",
]


def flag_keyword_match(descriptions: List[str], keywords: List[str]) -> np.ndarray:
    """Flags which condition descriptions mention at least one keyword.

    A patient is flagged (True) if their condition text contains ANY of the
    given keywords as a whole word or phrase, case-insensitively.

    "Whole word or phrase" matters: the keyword "pain" should match
    "lower back pain" but NOT "painless skin tag removal" (the substring
    "pain" appears in "painless", but not as a standalone word). Likewise
    the keyword "back pain" should only match when that exact two-word
    phrase appears — "back to back appointments" should NOT match.

    Hint: build a single regular expression of the form
    r"\\b(?:keyword1|keyword2|...)\\b" (escape each keyword with
    re.escape so punctuation in a keyword can't break the pattern), then
    use re.search against each lower-cased description. This should take
    about 4-6 lines.

    :param descriptions: A list of N condition-text strings (one per
    patient), e.g. "chronic kidney disease; type 2 diabetes".
    :param keywords: The list of keywords/phrases to search for.
    :return: A length-N boolean NumPy array; True where a keyword matched.
    """
    raise NotImplementedError("TODO: implement flag_keyword_match")


def standardize_features(
    feature_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PROVIDED — you do not need to edit this function.

    Z-scores each column of feature_matrix (subtract the column mean,
    divide by the column standard deviation) so that gradient descent in
    train_logistic_regression converges in a reasonable number of
    iterations regardless of each feature's original scale (age in years
    vs. healthcare expenses in dollars, for example).

    :param feature_matrix: A 2-D array of shape (N, D).
    :return: A tuple (scaled_matrix, means, stds).
    """
    means = feature_matrix.mean(axis=0)
    stds = feature_matrix.std(axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    return (feature_matrix - means) / stds, means, stds


def stratified_split(labels: np.ndarray, val_fraction: float, seed: int) -> np.ndarray:
    """Creates a stratified train/validation split, by hand (no sklearn).

    A plain random split can accidentally put almost all of a rare class
    into training and almost none into validation. A stratified split
    avoids this by taking the same fraction of examples from each class,
    so the class balance in train and validation both match the overall
    class balance.

    Follow these exact steps so your result is reproducible and matches
    the autograder:
      1. Create ``rng = np.random.default_rng(seed)``.
      2. Process the sorted unique class labels in ascending order
         (e.g. class 0 before class 1).
      3. For each class, find the indices of `labels` belonging to that
         class (in their original order), then shuffle just those indices
         with ``rng.permutation(...)``.
      4. The number of validation examples for that class is
         ``round(val_fraction * number_of_examples_in_that_class)``.
      5. The first that-many shuffled indices for the class go to
         validation; mark them True in the result. Everything else is
         False (i.e., train).

    About 6-10 lines.

    :param labels: A length-N array of 0/1 class labels.
    :param val_fraction: Fraction of each class to reserve for validation,
    e.g. 0.2 for 20%.
    :param seed: Random seed, for reproducibility.
    :return: A length-N boolean array; True marks a validation example.
    """
    raise NotImplementedError("TODO: implement stratified_split")


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Computes the logistic sigmoid, element-wise: 1 / (1 + e^-z).

    1 line.

    :param z: An array of any shape.
    :return: An array of the same shape, with every value in (0, 1).
    """
    raise NotImplementedError("TODO: implement sigmoid")


def predict_proba(
    feature_matrix: np.ndarray, weights: np.ndarray, bias: float
) -> np.ndarray:
    """PROVIDED — you do not need to edit this function.

    Computes the predicted probability of the positive class for each row
    of feature_matrix, using your sigmoid() function.

    :param feature_matrix: A 2-D array of shape (N, D).
    :param weights: A length-D array.
    :param bias: A scalar.
    :return: A length-N array of probabilities in (0, 1).
    """
    return sigmoid(feature_matrix.dot(weights) + bias)


def logistic_regression_gradients(
    feature_matrix: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
) -> Tuple[np.ndarray, float]:
    """Computes the gradient of the log-loss with respect to the weights
    and bias of a logistic regression model.

    First compute the model's current predictions using predict_proba function,
    then the error between predictions and the true labels y:

        error = predictions - y

    The gradient of the weights is the (feature-weighted) average error
    over all N examples:

        grad_weights = (feature_matrix^T . error) / N

    The gradient of the bias is just the average error:

        grad_bias = mean(error)

    About 3-4 lines.

    :param feature_matrix: A 2-D array of shape (N, D) of input features.
    :param y: A length-N array of 0/1 true labels.
    :param weights: The model's current length-D weight vector.
    :param bias: The model's current bias term (a scalar).
    :return: A tuple (grad_weights, grad_bias).
    """
    raise NotImplementedError("TODO: implement logistic_regression_gradients")


def train_logistic_regression(
    feature_matrix: np.ndarray,
    y: np.ndarray,
    iterations: int,
    learning_rate: float,
) -> Tuple[np.ndarray, float]:
    """Trains a logistic regression model with gradient descent.

    Start with weights of all zeros (length D) and bias 0.0. Then,
    `iterations` times: compute the gradients with
    logistic_regression_gradients, and update both weights and bias by
    subtracting the learning rate times their respective gradients.

    About 5-7 lines.

    :param feature_matrix: A 2-D array of shape (N, D) of (already
    standardized) input features.
    :param y: A length-N array of 0/1 true labels.
    :param iterations: Number of gradient descent steps to take.
    :param learning_rate: Step size for each gradient descent update.
    :return: A tuple (weights, bias) — the trained parameters.
    """
    raise NotImplementedError("TODO: implement train_logistic_regression")


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Counts true positives, false positives, true negatives, and false
    negatives between predicted and true binary labels.

    About 4 lines.

    :param y_true: A length-N array of 0/1 true labels.
    :param y_pred: A length-N array of 0/1 predicted labels.
    :return: A tuple of plain Python ints (tp, fp, tn, fn).
    """
    raise NotImplementedError("TODO: implement confusion_counts")


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Computes precision, recall, and F1-score from confusion counts.

        precision = tp / (tp + fp)
        recall    = tp / (tp + fn)
        f1        = 2 * precision * recall / (precision + recall)

    If a denominator would be zero, return 0.0 for that quantity instead
    of dividing by zero.

    About 5-8 lines.

    :param tp: Number of true positives.
    :param fp: Number of false positives.
    :param fn: Number of false negatives.
    :return: A tuple (precision, recall, f1).
    """
    raise NotImplementedError("TODO: implement precision_recall_f1")


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Computes the area under the ROC curve (AUC), from scratch.

    Use the pairwise-comparison definition of AUC (equivalent to the
    Mann-Whitney U statistic): consider every possible (positive example,
    negative example) pair. AUC is the fraction of those pairs where the
    positive example's score is higher than the negative example's score
    (count a tie as half a point):

        AUC = (sum over all pos/neg pairs of:
                  1   if score_pos >  score_neg
                  0.5 if score_pos == score_neg
                  0   if score_pos <  score_neg)
              / (number_of_positive_examples * number_of_negative_examples)

    You may assume y_true contains at least one 0 and at least one 1.
    A double loop over positives and negatives is fine at this dataset
    size — clarity matters more than speed here. About 6-10 lines.

    :param y_true: A length-N array of 0/1 true labels.
    :param scores: A length-N array of predicted scores/probabilities
    (higher means more likely positive).
    :return: The AUC, a float between 0 and 1.
    """
    raise NotImplementedError("TODO: implement roc_auc")


# =============================================================================
#
# Local verification  ─  run:  python pipeline.py
#
# Do not modify anything below this line.
#
# =============================================================================

_FEATURE_COLS = [
    "age", "is_female", "number_of_unique_meds", "number_of_encounters",
    "number_of_procedures", "unique_procedures", "pain_severity",
    "body_height", "body_weight", "body_mass_index",
    "systolic_blood_pressure", "diastolic_blood_pressure",
    "heart_rate", "respiratory_rate",
    "qaly", "daly", "qols", "healthcare_expenses", "healthcare_coverage",
]

def _run_metrics(data_path, seed, lr, steps, vf): 
    """Run the full pipeline and return a metrics dict. Not part of the graded API."""
    df = pd.read_csv(data_path)
    labels = flag_keyword_match(df["condition_text"].tolist(), PAIN_KEYWORDS).astype(float)
    x_all = df[_FEATURE_COLS].values.astype(float)

    is_val = stratified_split(labels.astype(int), vf, seed)
    x_train, y_train = x_all[~is_val], labels[~is_val]
    x_val, y_val     = x_all[is_val],  labels[is_val]

    x_train_s, means, stds = standardize_features(x_train)
    x_val_s = (x_val - means) / stds

    weights, bias = train_logistic_regression(
        x_train_s, y_train, iterations=int(steps), learning_rate=float(lr)
    )
    probs  = predict_proba(x_val_s, weights, bias)
    y_pred = (probs >= 0.5).astype(int)

    tp_v, fp_v, tn_v, fn_v = confusion_counts(y_val.astype(int), y_pred)
    _, _, f1_v = precision_recall_f1(tp_v, fp_v, fn_v)
    auc_v = roc_auc(y_val.astype(int), probs)
    acc_v = (tp_v + tn_v) / len(y_val)
    train_probs = predict_proba(x_train_s, weights, bias)
    eps = 1e-9
    loss_v = float(-np.mean(
        y_train * np.log(train_probs + eps) + (1 - y_train) * np.log(1 - train_probs + eps)
    ))
    return {"auc": auc_v, "accuracy": acc_v, "f1": f1_v, "loss": loss_v}


def _verify(): 
    """Local verification — called by __main__. Not part of the graded API."""
    root = Path(__file__).parent.parent   # assignment-1/

    p = get_sandbox_params()
    student_id = p.get("student_id", "").strip()
    lr, steps, vf = p["learning_rate"], p["steps"], p["val_fraction"]
    if not student_id:
        print("\n  Fill in 'student_id' in get_sandbox_params() with your NetID.\n")
        raise SystemExit(1)
    if lr == 0 or steps == 0:
        print("\n  Fill in 'learning_rate' and 'steps' in get_sandbox_params()\n"
              "  with the values from the UI Training Explorer.\n")
        raise SystemExit(1)

    # Derive the same seed used by the UI explorer (deterministic from student_id)
    h_val = int(hashlib.sha256(student_id.lower().encode()).hexdigest(), 16)
    seed = h_val % 900 + 100

    print(f"\n  Student : {student_id}   Seed: {seed}")
    print(f"  Params  : lr={lr}, steps={steps}, val_fraction={vf}\n")

    m = _run_metrics(root / "data" / "patient_features.csv", seed, lr, steps, vf)

    print("  Metric    Value")
    print("  ------------------")
    print(f"  {'F1':<9} {m['f1']:.3f}")
    print(f"  {'Accuracy':<9} {m['accuracy']:.3f}")
    print(f"  {'AUC':<9} {m['auc']:.3f}")
    print(f"  {'Loss':<9} {m['loss']:.3f}")
    print()
    print("  Note: This local output is informational only and does not reveal hidden test code.")
    print("  The official grader will perform the final evaluation.")
    print("  Run 'pytest test_pipeline.py -v' to verify your implementation passes all tests.")
    print("  Submit your pipeline.py to GitHub for grading.")


if __name__ == "__main__":
    _verify()

