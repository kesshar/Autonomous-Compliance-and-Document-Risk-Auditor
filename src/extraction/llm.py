"""Configurable LLM-based contract clause extractor with environment variable configuration."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple
from src.extraction.interface import BaseExtractor
from src.extraction.schema import ExtractionResult, ExtractionStatus

logger = logging.getLogger(__name__)


class LLMExtractor(BaseExtractor):
    """Configurable LLM extractor.

    Supported providers via environment variables:
    - LLM_PROVIDER: 'openai', 'anthropic', 'gemini', 'azure'
    - OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY
    - LLM_MODEL: model name (e.g., 'gpt-4o', 'claude-3-5-sonnet', 'gemini-1.5-pro')

    Strict constraints:
    - Never hardcodes API keys.
    - If the LLM provider is unconfigured, refuses to run and reports clear configuration status
      instead of inventing predictions.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "")).lower()
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

        # Resolve API key from provider-specific env vars
        self.api_key = api_key
        if not self.api_key:
            if self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider in {"gemini", "google"}:
                self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        self.is_configured = bool(self.provider and self.api_key)

    def check_configuration(self) -> Tuple[bool, str]:
        """Validates whether required environment variables and credentials are present."""
        if not self.provider:
            return False, "LLM_PROVIDER environment variable is not set (e.g. 'openai', 'anthropic', 'gemini')."
        if not self.api_key:
            return False, f"API key for provider '{self.provider}' is not set in environment."
        return True, f"Configured for provider '{self.provider}' with model '{self.model}'."

    def extract(
        self,
        contract_text: str,
        category_name: str,
        category_description: str,
        source_document_identifier: Optional[str] = None,
    ) -> ExtractionResult:
        is_ok, msg = self.check_configuration()
        if not is_ok:
            raise RuntimeError(
                f"Model evaluation cannot run because LLM provider is not configured: {msg} "
                "Configure your environment or use RuleBasedBaselineExtractor."
            )

        # In production or when credentials exist, this calls the LLM API and parses
        # structured JSON output into ExtractionResult.
        # For Milestone 1, placeholder for live network call with schema validation:
        return self._call_llm_api(
            contract_text=contract_text,
            category_name=category_name,
            category_description=category_description,
            source_document_identifier=source_document_identifier,
        )

    def _call_llm_api(
        self,
        contract_text: str,
        category_name: str,
        category_description: str,
        source_document_identifier: Optional[str] = None,
    ) -> ExtractionResult:
        # Live provider integration hook
        raise NotImplementedError(
            f"Active provider '{self.provider}' client call requires active network session."
        )

