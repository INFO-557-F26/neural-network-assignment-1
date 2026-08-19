import importlib
import math
import os
import numpy as np
import pytest

pipeline = importlib.import_module(os.environ.get("PIPELINE_IMPL", "pipeline"))

def s(x):
    return 1 / (1 + math.exp(-x))

@pytest.mark.timeout(2)
def test_flag_keyword_match():
    descriptions = [
        "chronic kidney disease",          # "chronic" is a keyword -> True
        "painless skin tag removal",       # "pain" is NOT a whole word here -> False
        "back to back appointments scheduled",  # no "back pain" phrase -> False
        "lower back pain",                 # "back pain" phrase present -> True
        "type 2 diabetes mellitus",        # no keyword -> False
        "Rheumatoid Arthritis",            # case-insensitive -> True
    ]
    flags = pipeline.flag_keyword_match(descriptions, pipeline.PAIN_KEYWORDS)
    np.testing.assert_array_equal(
        flags, np.array([True, False, False, True, False, True])
    )


@pytest.mark.timeout(2)
def test_stratified_split():
    # 8 examples of class 0, 4 examples of class 1
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    is_val = pipeline.stratified_split(labels, val_fraction=0.25, seed=42)

    assert is_val.dtype == np.bool_
    assert is_val.shape == labels.shape

    expected = np.array([
        False, False, False, True, True, False, False, False,
        False, False, False, True,
    ])
    np.testing.assert_array_equal(is_val, expected)

    # Stratification property: each class contributes the same fraction.
    for cls in (0, 1):
        class_mask = labels == cls
        frac_val = is_val[class_mask].mean()
        assert math.isclose(frac_val, 0.25, abs_tol=1e-9)

    # Reproducibility: same seed -> same split.
    again = pipeline.stratified_split(labels, val_fraction=0.25, seed=42)
    np.testing.assert_array_equal(is_val, again)


@pytest.mark.timeout(2)
def test_sigmoid():
    result = pipeline.sigmoid(np.array([-100.0, 0.0, 100.0]))
    np.testing.assert_allclose(result, np.array([s(-100.0), s(0.0), s(100.0)]),
                                atol=1e-12)


@pytest.mark.timeout(2)
def test_logistic_regression_gradients():
    X = np.array([[0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    weights = np.array([0.1, -0.2])
    bias = 0.05

    [p1, p2, p3, p4] = [s(0.1 * x1 + -0.2 * x2 + bias) for x1, x2 in X]
    errors = np.array([p1 - 0, p2 - 0, p3 - 1, p4 - 1])

    expected_grad_w = np.array([
        (0 * errors[0] + 1 * errors[1] + 0 * errors[2] + 1 * errors[3]) / 4,
        (0 * errors[0] + 0 * errors[1] + 1 * errors[2] + 1 * errors[3]) / 4,
    ])
    expected_grad_b = errors.mean()

    grad_w, grad_b = pipeline.logistic_regression_gradients(X, y, weights, bias)
    np.testing.assert_allclose(grad_w, expected_grad_w, atol=1e-10)
    assert math.isclose(grad_b, expected_grad_b, abs_tol=1e-10)


@pytest.mark.timeout(2)
def test_train_logistic_regression_separable():
    # A trivially separable 1-D problem: negative inputs are class 0,
    # positive inputs are class 1.
    X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 0, 1, 1, 1])

    weights, bias = pipeline.train_logistic_regression(
        X, y, iterations=500, learning_rate=0.5
    )
    probs = pipeline.predict_proba(X, weights, bias)
    preds = (probs >= 0.5).astype(int)
    assert np.all(preds == y)


@pytest.mark.timeout(2)
def test_confusion_counts():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    tp, fp, tn, fn = pipeline.confusion_counts(y_true, y_pred)
    assert (tp, fp, tn, fn) == (2, 0, 2, 1)


@pytest.mark.timeout(2)
def test_precision_recall_f1():
    precision, recall, f1 = pipeline.precision_recall_f1(tp=2, fp=0, fn=1)
    assert math.isclose(precision, 1.0, abs_tol=1e-9)
    assert math.isclose(recall, 2 / 3, abs_tol=1e-9)
    assert math.isclose(f1, 0.8, abs_tol=1e-9)

    # Degenerate case: no positives predicted and none true -> no
    # division by zero, everything should come back as 0.0.
    precision0, recall0, f10 = pipeline.precision_recall_f1(tp=0, fp=0, fn=0)
    assert (precision0, recall0, f10) == (0.0, 0.0, 0.0)


@pytest.mark.timeout(2)
def test_roc_auc_perfect_separation():
    y_true = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    auc = pipeline.roc_auc(y_true, scores)
    assert math.isclose(auc, 1.0, abs_tol=1e-9)


@pytest.mark.timeout(2)
def test_roc_auc_with_tie():
    y_true = np.array([0, 1, 0, 1])
    scores = np.array([0.3, 0.3, 0.6, 0.9])
    # positives: 0.3, 0.9 | negatives: 0.3, 0.6
    # (0.3 vs 0.3)=0.5  (0.3 vs 0.6)=0  (0.9 vs 0.3)=1  (0.9 vs 0.6)=1
    # AUC = (0.5 + 0 + 1 + 1) / 4 = 0.625
    auc = pipeline.roc_auc(y_true, scores)
    assert math.isclose(auc, 0.625, abs_tol=1e-9)


@pytest.mark.timeout(2)
def test_get_sandbox_params_filled_in():
    """get_sandbox_params() must be filled in 
    before submitting. The starter values (empty student_id, learning_rate
    or steps of 0) are placeholders, not a valid answer, and will fail this
    test on purpose until you replace them."""
    params = pipeline.get_sandbox_params()

    student_id = params.get("student_id", "")
    assert isinstance(student_id, str) and student_id.strip(), (
        "get_sandbox_params()['student_id'] is empty. Fill in your NetID."
    )
    assert params.get("learning_rate", 0) > 0, (
        "get_sandbox_params()['learning_rate'] is 0. Fill in the value you "
        "found in the UI Training Explorer."
    )
    assert params.get("steps", 0) > 0, (
        "get_sandbox_params()['steps'] is 0. Fill in the value you found "
        "in the UI Training Explorer."
    )

