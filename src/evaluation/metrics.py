"""Mathematical metric computations for legal clause extraction evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from src.extraction.normalizer import (
    check_exact_match,
    compute_token_jaccard,
    normalize_text,
)


class CategoryMetrics(BaseModel):
    """Evaluation metrics for a single contractual category."""

    category: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    exact_match_ratio: float = 0.0
    avg_jaccard_iou: float = 0.0
    total_gold_instances: int = 0
    total_predictions: int = 0


class AggregateMetrics(BaseModel):
    """Aggregate macro and micro metrics across all categories."""

    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    micro_precision: float = 0.0
    micro_recall: float = 0.0
    micro_f1: float = 0.0
    total_samples: int = 0
    total_categories_evaluated: int = 0


def calculate_precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Computes precision, recall, and F1 given TP, FP, FN counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_evidence_span_overlap(
    gold_span_text: str,
    pred_span_text: str,
    iou_threshold: float = 0.5,
    category: Optional[str] = None,
) -> Dict[str, float]:
    """Calculates Jaccard word-level overlap and determines if threshold is satisfied."""
    jaccard = compute_token_jaccard(gold_span_text, pred_span_text)
    exact_match = 1.0 if check_exact_match(gold_span_text, pred_span_text, category) else 0.0
    is_match = jaccard >= iou_threshold or exact_match == 1.0
    return {
        "jaccard": round(jaccard, 4),
        "exact_match": exact_match,
        "is_match": 1.0 if is_match else 0.0,
    }


def compute_macro_average(category_metrics_list: List[CategoryMetrics]) -> Dict[str, float]:
    """Calculates unweighted macro-averaged precision, recall, and F1 across categories."""
    if not category_metrics_list:
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    valid_cats = [m for m in category_metrics_list if (m.total_gold_instances > 0 or m.total_predictions > 0)]
    if not valid_cats:
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    macro_p = sum(m.precision for m in valid_cats) / len(valid_cats)
    macro_r = sum(m.recall for m in valid_cats) / len(valid_cats)
    macro_f1 = sum(m.f1 for m in valid_cats) / len(valid_cats)

    return {
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
    }


def compute_micro_average(category_metrics_list: List[CategoryMetrics]) -> Dict[str, float]:
    """Calculates globally pooled micro-averaged precision, recall, and F1."""
    total_tp = sum(m.true_positives for m in category_metrics_list)
    total_fp = sum(m.false_positives for m in category_metrics_list)
    total_fn = sum(m.false_negatives for m in category_metrics_list)

    prf = calculate_precision_recall_f1(total_tp, total_fp, total_fn)
    return {
        "micro_precision": prf["precision"],
        "micro_recall": prf["recall"],
        "micro_f1": prf["f1"],
    }

