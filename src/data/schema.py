"""Data models and schemas for contracts, annotations, and dataset manifests."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class EvidenceSpan(BaseModel):
    """Represents an exact text span of evidence in a contract."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(..., description="Exact textual content of the evidence span.")
    start_char: int = Field(..., ge=0, description="Character-level starting offset in the contract text.")
    end_char: int = Field(..., ge=0, description="Character-level ending offset in the contract text.")

    def length(self) -> int:
        """Return the length of the evidence span in characters."""
        return max(0, self.end_char - self.start_char)


class ClauseAnnotation(BaseModel):
    """Represents the human ground-truth annotation for a single clause category in a contract."""

    contract_id: str = Field(..., description="Unique contract identifier.")
    category_name: str = Field(..., description="Canonical CUAD category name.")
    is_present: bool = Field(..., description="True if the clause is present in the contract.")
    answers: List[str] = Field(default_factory=list, description="Extracted answer values / texts.")
    evidence_spans: List[EvidenceSpan] = Field(
        default_factory=list,
        description="Character spans in the document where evidence was annotated.",
    )
    raw_question: Optional[str] = Field(None, description="Original question prompt from dataset.")


class ContractDocument(BaseModel):
    """Represents a complete contract document with text and optional ground truth annotations."""

    id: str = Field(..., description="Unique document ID (e.g., contract title).")
    title: str = Field(..., description="Contract title / filename.")
    text: str = Field(..., description="Full text context of the contract agreement.")
    annotations: Dict[str, ClauseAnnotation] = Field(
        default_factory=dict,
        description="Mapping from category_name to ClauseAnnotation.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary document metadata (e.g. split, word count, filing date).",
    )

    def get_unlabeled_context(self) -> Dict[str, Any]:
        """Provides an inference-safe view of the contract text with ZERO gold annotations.

        Guarantees that models evaluated against the contract do not receive
        answers or evidence spans during inference.
        """
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "length_chars": len(self.text),
        }

    def get_annotation(self, category_name: str) -> Optional[ClauseAnnotation]:
        """Retrieve annotation for a given category name."""
        return self.annotations.get(category_name)


class FileChecksum(BaseModel):
    """Metadata for a downloaded or processed file."""

    path: str
    size_bytes: int
    sha256: str


class DatasetManifest(BaseModel):
    """Manifest recording provenance, integrity, license, and attribution for the dataset."""

    dataset_name: str = "Contract Understanding Atticus Dataset (CUAD)"
    dataset_version: str = "v1"
    source_url: str
    license: str = "CC BY 4.0 (Annotations); Underlying contracts are public SEC EDGAR filings"
    download_timestamp: str
    sha256_checksum: str
    attribution: str = (
        "Hendrycks et al., 'CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review', "
        "NeurIPS 2021. Curated by The Atticus Project (https://www.atticusprojectai.org/cuad)."
    )
    rights_limitation: str = (
        "Underlying contract documents originate from public SEC EDGAR filings. "
        "The Atticus Project does not hold underlying commercial rights to the contracts themselves; "
        "annotations are provided under CC BY 4.0 for research and compliance purposes."
    )
    total_documents: int = 0
    files: Dict[str, FileChecksum] = Field(default_factory=dict)

