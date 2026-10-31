"""Abstract extraction interface for legal contract attribute extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from src.extraction.schema import ExtractionResult


class BaseExtractor(ABC):
    """Abstract base class for all contractual clause extraction engines.

    Guarantees:
    - Never exposes gold annotations during inference.
    - Returns structured Pydantic ExtractionResult objects.
    - Employs NOT_FOUND status rather than hallucinating missing attributes.
    """

    @abstractmethod
    def extract(
        self,
        contract_text: str,
        category_name: str,
        category_description: str,
        source_document_identifier: Optional[str] = None,
    ) -> ExtractionResult:
        """Extracts a contractual attribute from contract text.

        Args:
            contract_text: The full verbatim text of the contract.
            category_name: Canonical CUAD category name.
            category_description: Formal definition / instructions for the category.
            source_document_identifier: Optional contract title or filename.

        Returns:
            Structured ExtractionResult.
        """
        pass

    def batch_extract(
        self,
        contract_text: str,
        categories: List[Dict[str, str]],
        source_document_identifier: Optional[str] = None,
    ) -> List[ExtractionResult]:
        """Performs extraction for multiple categories across a contract.

        Args:
            contract_text: The full verbatim text of the contract.
            categories: List of dicts with 'name' and 'description' keys.
            source_document_identifier: Optional contract title or filename.

        Returns:
            List of ExtractionResult objects.
        """
        results: List[ExtractionResult] = []
        for cat in categories:
            res = self.extract(
                contract_text=contract_text,
                category_name=cat["name"],
                category_description=cat.get("description", ""),
                source_document_identifier=source_document_identifier,
            )
            results.append(res)
        return results

