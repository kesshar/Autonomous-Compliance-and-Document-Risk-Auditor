"""Contract clause extraction interfaces, baseline models, schemas, and normalizers."""

from src.extraction.schema import ExtractionResult, ExtractionStatus
from src.extraction.interface import BaseExtractor
from src.extraction.baseline import RuleBasedBaselineExtractor
from src.extraction.llm import LLMExtractor
from src.extraction.normalizer import (
    normalize_text,
    normalize_date,
    normalize_party,
    normalize_boolean,
    compute_token_jaccard,
    check_exact_match,
)

__all__ = [
    "ExtractionResult",
    "ExtractionStatus",
    "BaseExtractor",
    "RuleBasedBaselineExtractor",
    "LLMExtractor",
    "normalize_text",
    "normalize_date",
    "normalize_party",
    "normalize_boolean",
    "compute_token_jaccard",
    "check_exact_match",
]

