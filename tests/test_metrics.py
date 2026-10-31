"""Unit tests for precision, recall, F1, and span quality metrics."""

from src.evaluation.metrics import (
    CategoryMetrics,
    calculate_precision_recall_f1,
    compute_evidence_span_overlap,
    compute_macro_average,
    compute_micro_average,
)


def test_calculate_precision_recall_f1():
    # Standard balanced case: TP=8, FP=2, FN=2
    # Precision = 8/10 = 0.8
    # Recall = 8/10 = 0.8
    # F1 = 0.8
    res = calculate_precision_recall_f1(tp=8, fp=2, fn=2)
    assert res["precision"] == 0.8
    assert res["recall"] == 0.8
    assert res["f1"] == 0.8

    # Zero positive predictions: TP=0, FP=0, FN=5
    res_zero_pred = calculate_precision_recall_f1(tp=0, fp=0, fn=5)
    assert res_zero_pred["precision"] == 0.0
    assert res_zero_pred["recall"] == 0.0
    assert res_zero_pred["f1"] == 0.0

    # Zero instances total: TP=0, FP=0, FN=0
    res_zero = calculate_precision_recall_f1(tp=0, fp=0, fn=0)
    assert res_zero["precision"] == 0.0
    assert res_zero["recall"] == 0.0
    assert res_zero["f1"] == 0.0


def test_compute_macro_and_micro_average():
    c1 = CategoryMetrics(
        category="Agreement Date",
        true_positives=10,
        false_positives=0,
        false_negatives=0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        total_gold_instances=10,
        total_predictions=10,
    )
    c2 = CategoryMetrics(
        category="Governing Law",
        true_positives=5,
        false_positives=5,
        false_negatives=5,
        precision=0.5,
        recall=0.5,
        f1=0.5,
        total_gold_instances=10,
        total_predictions=10,
    )

    macro = compute_macro_average([c1, c2])
    assert macro["macro_precision"] == 0.75
    assert macro["macro_recall"] == 0.75
    assert macro["macro_f1"] == 0.75

    # Micro: total TP=15, total FP=5, total FN=5
    # Micro Precision = 15/20 = 0.75
    # Micro Recall = 15/20 = 0.75
    micro = compute_micro_average([c1, c2])
    assert micro["micro_precision"] == 0.75
    assert micro["micro_recall"] == 0.75
    assert micro["micro_f1"] == 0.75


def test_compute_evidence_span_overlap():
    gold = "This Agreement shall be governed by the laws of New York."
    pred = "governed by the laws of New York"
    res = compute_evidence_span_overlap(gold, pred, iou_threshold=0.5)

    assert res["jaccard"] >= 0.5
    assert res["is_match"] == 1.0

    unrelated = "Entire agreement clause"
    res_bad = compute_evidence_span_overlap(gold, unrelated, iou_threshold=0.5)
    assert res_bad["jaccard"] < 0.2
    assert res_bad["is_match"] == 0.0

