"""Unit tests for the extraction evaluation engine and reporting pipeline."""

from pathlib import Path
from src.data.loader import CUADLoader
from src.evaluation.evaluator import ExtractionEvaluator
from src.extraction.baseline import RuleBasedBaselineExtractor


def test_evaluator_end_to_end(mini_squad_file: Path, tmp_path: Path):
    loader = CUADLoader()
    docs = loader.parse_squad_json(mini_squad_file)

    extractor = RuleBasedBaselineExtractor()
    evaluator = ExtractionEvaluator(
        extractor=extractor,
        iou_threshold=0.5,
        max_error_examples=5,
    )

    target_categories = ["Parties", "Agreement Date", "Governing Law", "Liquidated Damages"]
    report = evaluator.evaluate_documents(docs, target_categories)

    assert report.total_documents_evaluated == 2
    assert len(report.categories) == 4

    # Agreement Date was in contract 1
    m_date = report.categories["Agreement Date"]
    assert m_date.true_positives >= 1
    assert m_date.precision > 0.0

    # Macro & micro metrics are computed
    assert report.aggregate_metrics.total_categories_evaluated == 4
    assert 0.0 <= report.aggregate_metrics.macro_f1 <= 1.0
    assert 0.0 <= report.aggregate_metrics.micro_f1 <= 1.0

    # Error examples exist in report structure
    assert "Agreement Date" in report.error_examples
    assert "Liquidated Damages" in report.error_examples

    # Test saving reports to disk
    paths = evaluator.save_reports(report, output_dir=tmp_path, prefix="test_run")
    assert paths["json_report"].is_file()
    assert paths["csv_summary"].is_file()
    assert paths["error_examples"].is_file()

    # Validate CSV content
    with open(paths["csv_summary"], "r", encoding="utf-8") as f:
        csv_text = f.read()
        assert "Category,Precision,Recall,F1-Score" in csv_text
        assert "Agreement Date" in csv_text
        assert "--- MACRO AVERAGE ---" in csv_text
        assert "--- MICRO AVERAGE ---" in csv_text

