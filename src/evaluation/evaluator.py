"""Evaluation engine for contractual attribute extraction benchmarking."""

from __future__ import annotations

import csv
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from src.data.loader import load_category_aliases
from src.data.schema import ContractDocument
from src.evaluation.metrics import (
    AggregateMetrics,
    CategoryMetrics,
    calculate_precision_recall_f1,
    compute_macro_average,
    compute_micro_average,
)
from src.extraction.interface import BaseExtractor
from src.extraction.normalizer import (
    check_exact_match,
    compute_token_jaccard,
    normalize_text,
)
from src.extraction.schema import ExtractionResult, ExtractionStatus

logger = logging.getLogger(__name__)


class ErrorExamples(BaseModel):
    """Categorized extraction error examples for diagnostic review."""

    false_positives: List[Dict[str, Any]] = Field(default_factory=list)
    false_negatives: List[Dict[str, Any]] = Field(default_factory=list)
    incorrect_extractions: List[Dict[str, Any]] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    """Complete evaluation report containing category metrics, aggregates, and error cases."""

    timestamp: str
    total_documents_evaluated: int
    categories: Dict[str, CategoryMetrics]
    aggregate_metrics: AggregateMetrics
    error_examples: Dict[str, ErrorExamples]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractionEvaluator:
    """Evaluates an extraction engine against CUAD ground truth annotations."""

    def __init__(
        self,
        extractor: BaseExtractor,
        category_config_path: Union[str, Path] = "configs/cuad_categories.yaml",
        iou_threshold: float = 0.5,
        max_error_examples: int = 5,
    ):
        self.extractor = extractor
        self.iou_threshold = iou_threshold
        self.max_error_examples = max_error_examples

        # Load category metadata
        self.category_descriptions: Dict[str, str] = {}
        config_p = Path(category_config_path)
        if config_p.is_file():
            import yaml
            with open(config_p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                for c in data.get("categories", []):
                    self.category_descriptions[c["name"]] = c.get("description", "")

    def evaluate_documents(
        self,
        documents: Dict[str, ContractDocument],
        target_categories: List[str],
    ) -> EvaluationReport:
        """Runs the extraction evaluation across documents for the given categories."""
        logger.info(
            "Evaluating %d documents across %d categories (IoU threshold: %.2f)...",
            len(documents),
            len(target_categories),
            self.iou_threshold,
        )

        cat_metrics_map: Dict[str, CategoryMetrics] = {}
        error_examples_map: Dict[str, ErrorExamples] = {}

        for category_name in target_categories:
            cat_desc = self.category_descriptions.get(category_name, "")
            cm = self._evaluate_single_category(
                documents=documents,
                category_name=category_name,
                category_description=cat_desc,
            )
            cat_metrics_map[category_name] = cm["metrics"]
            error_examples_map[category_name] = cm["errors"]

        metrics_list = list(cat_metrics_map.values())
        macro = compute_macro_average(metrics_list)
        micro = compute_micro_average(metrics_list)

        agg = AggregateMetrics(
            macro_precision=macro["macro_precision"],
            macro_recall=macro["macro_recall"],
            macro_f1=macro["macro_f1"],
            micro_precision=micro["micro_precision"],
            micro_recall=micro["micro_recall"],
            micro_f1=micro["micro_f1"],
            total_samples=len(documents),
            total_categories_evaluated=len(target_categories),
        )

        return EvaluationReport(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            total_documents_evaluated=len(documents),
            categories=cat_metrics_map,
            aggregate_metrics=agg,
            error_examples=error_examples_map,
            metadata={"iou_threshold": self.iou_threshold},
        )

    def _evaluate_single_category(
        self,
        documents: Dict[str, ContractDocument],
        category_name: str,
        category_description: str,
    ) -> Dict[str, Any]:
        tp, fp, fn, tn = 0, 0, 0, 0
        exact_matches = 0
        jaccard_scores: List[float] = []

        fps_list: List[Dict[str, Any]] = []
        fns_list: List[Dict[str, Any]] = []
        incorrect_list: List[Dict[str, Any]] = []

        for doc_id, doc in documents.items():
            # Strictly get unlabeled context - prevent gold leakage during inference
            unlabeled = doc.get_unlabeled_context()
            pred = self.extractor.extract(
                contract_text=unlabeled["text"],
                category_name=category_name,
                category_description=category_description,
                source_document_identifier=doc_id,
            )

            gold = doc.get_annotation(category_name)
            gold_answers = gold.answers if gold else []
            gold_present = bool(gold and gold.is_present and gold_answers)

            pred_found = pred.is_found()
            pred_val = pred.extracted_value or ""

            if not gold_present:
                if pred_found:
                    fp += 1
                    if len(fps_list) < self.max_error_examples:
                        fps_list.append({
                            "contract_id": doc_id,
                            "category": category_name,
                            "predicted_value": pred_val,
                            "clause_snippet": (pred.clause_text or "")[:200],
                            "confidence": pred.confidence,
                        })
                else:
                    tn += 1
            else:
                # Gold IS present
                if not pred_found:
                    fn += 1
                    if len(fns_list) < self.max_error_examples:
                        fns_list.append({
                            "contract_id": doc_id,
                            "category": category_name,
                            "gold_answers": gold_answers[:3],
                            "reason": "Model returned NOT_FOUND or failed extraction",
                        })
                else:
                    # Model found something and Gold has answers
                    match_found = False
                    best_jaccard = 0.0
                    is_em = False

                    for ans in gold_answers:
                        # Compute token Jaccard & exact match
                        jaccard = compute_token_jaccard(ans, pred_val)
                        if jaccard > best_jaccard:
                            best_jaccard = jaccard
                        em = check_exact_match(ans, pred_val, category=category_name)
                        if em:
                            is_em = True

                        # For Parties or substring clauses, allow substring match
                        is_substr = False
                        if "parties" in category_name.lower():
                            norm_ans = normalize_text(ans)
                            norm_pred = normalize_text(pred_val)
                            is_substr = (norm_ans in norm_pred) or (norm_pred in norm_ans)

                        if jaccard >= self.iou_threshold or em or is_substr:
                            match_found = True
                            break

                    jaccard_scores.append(best_jaccard)
                    if is_em:
                        exact_matches += 1

                    if match_found:
                        tp += 1
                    else:
                        # Mismatched prediction
                        fn += 1
                        fp += 1
                        if len(incorrect_list) < self.max_error_examples:
                            incorrect_list.append({
                                "contract_id": doc_id,
                                "category": category_name,
                                "gold_answers": gold_answers[:2],
                                "predicted_value": pred_val,
                                "best_jaccard": round(best_jaccard, 4),
                            })

        prf = calculate_precision_recall_f1(tp, fp, fn)
        total_positives = tp + fn
        em_ratio = round(exact_matches / total_positives, 4) if total_positives > 0 else 0.0
        avg_jaccard = round(sum(jaccard_scores) / len(jaccard_scores), 4) if jaccard_scores else 0.0

        metrics = CategoryMetrics(
            category=category_name,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            precision=prf["precision"],
            recall=prf["recall"],
            f1=prf["f1"],
            exact_match_ratio=em_ratio,
            avg_jaccard_iou=avg_jaccard,
            total_gold_instances=total_positives,
            total_predictions=tp + fp,
        )

        errors = ErrorExamples(
            false_positives=fps_list,
            false_negatives=fns_list,
            incorrect_extractions=incorrect_list,
        )

        return {"metrics": metrics, "errors": errors}

    @staticmethod
    def save_reports(
        report: EvaluationReport,
        output_dir: Union[str, Path] = "reports/evaluation/cuad",
        prefix: str = "eval",
    ) -> Dict[str, Path]:
        """Saves evaluation results as machine-readable JSON and tabular CSV reports."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = out_dir / f"{prefix}_report_{timestamp_str}.json"
        csv_path = out_dir / f"{prefix}_summary_{timestamp_str}.csv"
        errors_path = out_dir / f"{prefix}_errors_{timestamp_str}.json"

        # 1. Full JSON Report
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)

        # 2. Detailed Error Examples JSON
        with open(errors_path, "w", encoding="utf-8") as f:
            errors_data = {
                cat: err.model_dump() for cat, err in report.error_examples.items()
            }
            json.dump(errors_data, f, indent=2)

        # 3. Summary CSV Report
        fieldnames = [
            "Category",
            "Precision",
            "Recall",
            "F1-Score",
            "Exact Match",
            "Avg Jaccard IoU",
            "TP",
            "FP",
            "FN",
            "TN",
            "Total Gold",
            "Total Predictions",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for cat_name, cm in report.categories.items():
                writer.writerow({
                    "Category": cat_name,
                    "Precision": f"{cm.precision:.4f}",
                    "Recall": f"{cm.recall:.4f}",
                    "F1-Score": f"{cm.f1:.4f}",
                    "Exact Match": f"{cm.exact_match_ratio:.4f}",
                    "Avg Jaccard IoU": f"{cm.avg_jaccard_iou:.4f}",
                    "TP": cm.true_positives,
                    "FP": cm.false_positives,
                    "FN": cm.false_negatives,
                    "TN": cm.true_negatives,
                    "Total Gold": cm.total_gold_instances,
                    "Total Predictions": cm.total_predictions,
                })
            # Add Macro and Micro Rows
            agg = report.aggregate_metrics
            writer.writerow({
                "Category": "--- MACRO AVERAGE ---",
                "Precision": f"{agg.macro_precision:.4f}",
                "Recall": f"{agg.macro_recall:.4f}",
                "F1-Score": f"{agg.macro_f1:.4f}",
                "Exact Match": "-",
                "Avg Jaccard IoU": "-",
                "TP": "-",
                "FP": "-",
                "FN": "-",
                "TN": "-",
                "Total Gold": "-",
                "Total Predictions": "-",
            })
            writer.writerow({
                "Category": "--- MICRO AVERAGE ---",
                "Precision": f"{agg.micro_precision:.4f}",
                "Recall": f"{agg.micro_recall:.4f}",
                "F1-Score": f"{agg.micro_f1:.4f}",
                "Exact Match": "-",
                "Avg Jaccard IoU": "-",
                "TP": "-",
                "FP": "-",
                "FN": "-",
                "TN": "-",
                "Total Gold": "-",
                "Total Predictions": "-",
            })

        logger.info("Saved JSON report to: %s", json_path)
        logger.info("Saved CSV summary to: %s", csv_path)
        logger.info("Saved error analysis to: %s", errors_path)

        return {
            "json_report": json_path,
            "csv_summary": csv_path,
            "error_examples": errors_path,
        }

