"""Pydantic schemas for contract extraction results and status."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ExtractionStatus(str, Enum):
    """Status of an extraction attempt."""

    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    UNCERTAIN = "UNCERTAIN"
    ERROR = "ERROR"


class ExtractionResult(BaseModel):
    """Structured Pydantic-validated representation of a clause extraction result."""

    model_config = ConfigDict(extra="ignore")

    category: str = Field(..., description="Canonical name of the contractual attribute/category.")
    extracted_value: Optional[str] = Field(
        None,
        description="The extracted value/answer (e.g. date, jurisdiction, parties list, or 'Yes'/'No'). "
        "Set to None or 'NOT_FOUND' if not present.",
    )
    clause_text: Optional[str] = Field(
        None,
        description="The relevant verbatim clause snippet located in the contract text.",
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="List of verbatim evidence text snippets from the contract supporting the extraction.",
    )
    confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Model confidence score between 0.0 and 1.0.",
    )
    status: ExtractionStatus = Field(
        default=ExtractionStatus.NOT_FOUND,
        description="Extraction status: FOUND, NOT_FOUND, UNCERTAIN, or ERROR.",
    )
    source_document_identifier: Optional[str] = Field(
        None,
        description="Unique identifier or filename of the source contract.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional diagnostic metadata (model name, latency, token count, etc.).",
    )

    @classmethod
    def not_found(
        cls,
        category: str,
        source_document_identifier: Optional[str] = None,
        confidence: float = 0.0,
        reason: Optional[str] = None,
    ) -> ExtractionResult:
        """Constructs an explicit NOT_FOUND result. Never invents missing data."""
        return cls(
            category=category,
            extracted_value="NOT_FOUND",
            clause_text=None,
            evidence=[],
            confidence=confidence,
            status=ExtractionStatus.NOT_FOUND,
            source_document_identifier=source_document_identifier,
            metadata={"reason": reason} if reason else {},
        )

    def is_found(self) -> bool:
        """Returns True if a clause was found and non-empty."""
        return (
            self.status == ExtractionStatus.FOUND
            and self.extracted_value is not None
            and self.extracted_value != "NOT_FOUND"
        )

