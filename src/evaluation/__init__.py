"""Evaluation metrics, benchmarking runners, and reporting tools."""

from src.evaluation.metrics import (
    CategoryMetrics,
    AggregateMetrics,
    calculate_precision_recall_f1,
    compute_evidence_span_overlap,
    compute_macro_average,
    compute_micro_average,
)
from src.evaluation.evaluator import (
    ExtractionEvaluator,
    EvaluationReport,
    ErrorExamples,
)

__all__ = [
    "CategoryMetrics",
    "AggregateMetrics",
    "calculate_precision_recall_f1",
    "compute_evidence_span_overlap",
    "compute_macro_average",
    "compute_micro_average",
    "ExtractionEvaluator",
    "EvaluationReport",
    "ErrorExamples",
]

