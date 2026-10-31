"""Unit tests for extraction interface, schemas, and extractor implementations."""

import pytest
from pydantic import ValidationError
from src.extraction.baseline import RuleBasedBaselineExtractor
from src.extraction.llm import LLMExtractor
from src.extraction.schema import ExtractionResult, ExtractionStatus


def test_extraction_result_validation():
    # Valid found result
    res = ExtractionResult(
        category="Governing Law",
        extracted_value="New York",
        clause_text="Governed by the laws of New York",
        evidence=["laws of New York"],
        confidence=0.9,
        status=ExtractionStatus.FOUND,
        source_document_identifier="CONTRACT_123",
    )
    assert res.category == "Governing Law"
    assert res.is_found() is True

    # Valid not_found factory
    nf = ExtractionResult.not_found(category="Audit Rights", source_document_identifier="CONTRACT_123")
    assert nf.status == ExtractionStatus.NOT_FOUND
    assert nf.extracted_value == "NOT_FOUND"
    assert nf.is_found() is False
    assert nf.confidence == 0.0


def test_confidence_score_bounds():
    with pytest.raises(ValidationError):
        ExtractionResult(category="Parties", confidence=1.5, status=ExtractionStatus.FOUND)

    with pytest.raises(ValidationError):
        ExtractionResult(category="Parties", confidence=-0.1, status=ExtractionStatus.FOUND)


def test_rule_based_baseline_extractor():
    extractor = RuleBasedBaselineExtractor()
    sample_text = (
        "This Agreement is dated September 7, 1999. "
        "Governed by the laws of the State of Delaware. "
        "Either party may terminate this agreement without cause upon 30 days notice."
    )

    # Date extraction
    r_date = extractor.extract(sample_text, "Agreement Date", "The date of the contract", "DOC_1")
    assert r_date.status == ExtractionStatus.FOUND
    assert "September 7, 1999" in r_date.extracted_value

    # Governing law extraction
    r_gov = extractor.extract(sample_text, "Governing Law", "Jurisdiction", "DOC_1")
    assert r_gov.status == ExtractionStatus.FOUND
    assert r_gov.extracted_value == "Delaware"

    # Termination for convenience extraction
    r_term = extractor.extract(sample_text, "Termination for Convenience", "Term without cause", "DOC_1")
    assert r_term.status == ExtractionStatus.FOUND
    assert r_term.extracted_value == "Yes"

    # Unmatched category returns NOT_FOUND (never invents missing data)
    r_audit = extractor.extract(sample_text, "Audit Rights", "Audit records", "DOC_1")
    assert r_audit.status == ExtractionStatus.NOT_FOUND
    assert r_audit.is_found() is False


def test_llm_extractor_unconfigured_error():
    # When no environment variables or API keys are set, LLMExtractor reports clear configuration status
    extractor = LLMExtractor(provider="", api_key="")
    is_ok, msg = extractor.check_configuration()
    assert is_ok is False
    assert "LLM_PROVIDER environment variable is not set" in msg

    # Calling extract raises RuntimeError rather than fabricating predictions
    with pytest.raises(RuntimeError, match="Model evaluation cannot run because LLM provider is not configured"):
        extractor.extract("Some text", "Governing Law", "Governing law description")

