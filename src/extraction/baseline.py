"""Deterministic rule-based baseline extractor for CUAD contract attributes."""

from __future__ import annotations

import re
from typing import Dict, List, Optional
from src.extraction.interface import BaseExtractor
from src.extraction.schema import ExtractionResult, ExtractionStatus

US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma",
    "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee",
    "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
    "England and Wales", "United Kingdom", "Canada"
]


class RuleBasedBaselineExtractor(BaseExtractor):
    """Deterministic, regex and heuristic-based extractor.

    Acts as an offline baseline for pipeline testing and evaluation when LLM APIs are not configured.
    Strictly avoids inventing contractual information.
    """

    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold

    def extract(
        self,
        contract_text: str,
        category_name: str,
        category_description: str,
        source_document_identifier: Optional[str] = None,
    ) -> ExtractionResult:
        if not contract_text or not contract_text.strip():
            return ExtractionResult.not_found(category_name, source_document_identifier)

        method_name = f"_extract_{category_name.lower().replace(' ', '_')}"
        extractor_fn = getattr(self, method_name, self._extract_generic)
        return extractor_fn(contract_text, category_name, source_document_identifier)

    def _extract_agreement_date(
        self, text: str, cat: str, doc_id: Optional[str]
    ) -> ExtractionResult:
        patterns = [
            r"(?:dated|entered into|made)\s+(?:as of\s+)?(?:this\s+)?(\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+[A-Za-z]+,?\s+\d{4})",
            r"(?:dated|as of)\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
            r"(?:dated|as of)\s+(\d{1,2}/\d{1,2}/\d{2,4})",
        ]
        return self._first_regex_match(text, cat, doc_id, patterns, confidence=0.85)

    def _extract_effective_date(
        self, text: str, cat: str, doc_id: Optional[str]
    ) -> ExtractionResult:
        patterns = [
            r"(?:effective\s+(?:date\s+)?(?:is|as of|on)\s+)([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
            r"(?:effective\s+as of\s+(?:this\s+)?\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+[A-Za-z]+,?\s+\d{4})",
            r"(?:effective\s+date\s*:\s*)([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
            r"(?:shall be effective immediately)",
        ]
        return self._first_regex_match(text, cat, doc_id, patterns, confidence=0.8)

    def _extract_governing_law(
        self, text: str, cat: str, doc_id: Optional[str]
    ) -> ExtractionResult:
        for state in US_STATES:
            pattern = rf"(?:governed by|construed in accordance with|laws of\s+(?:the\s+State\s+of\s+)?){re.escape(state)}"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                snippet_start = max(0, match.start() - 50)
                snippet_end = min(len(text), match.end() + 50)
                clause = text[snippet_start:snippet_end].strip()
                return ExtractionResult(
                    category=cat,
                    extracted_value=state,
                    clause_text=clause,
                    evidence=[match.group(0)],
                    confidence=0.85,
                    status=ExtractionStatus.FOUND,
                    source_document_identifier=doc_id,
                )
        return ExtractionResult.not_found(cat, doc_id)

    def _extract_termination_for_convenience(
        self, text: str, cat: str, doc_id: Optional[str]
    ) -> ExtractionResult:
        patterns = [
            r"(?:terminate\s+(?:this\s+agreement\s+)?(?:without\s+cause|for\s+convenience))",
            r"(?:terminate\s+at\s+any\s+time\s+upon\s+\d+\s+days\s+prior\s+written\s+notice)",
        ]
        return self._presence_regex_match(text, cat, doc_id, patterns, confidence=0.75)

    def _extract_anti_assignment(
        self, text: str, cat: str, doc_id: Optional[str]
    ) -> ExtractionResult:
        patterns = [
            r"(?:neither\s+party\s+may\s+assign\s+this\s+agreement)",
            r"(?:shall\s+not\s+assign\s+(?:or\s+transfer\s+)?(?:this\s+agreement|its\s+rights)\s+without\s+(?:the\s+)?prior\s+written\s+consent)",
        ]
        return self._presence_regex_match(text, cat, doc_id, patterns, confidence=0.8)

    def _extract_audit_rights(
        self, text: str, cat: str, doc_id: Optional[str]
    ) -> ExtractionResult:
        patterns = [
            r"(?:right\s+to\s+audit\s+(?:the\s+)?books\s+and\s+records)",
            r"(?:audit\s+(?:and\s+inspect\s+)?(?:the\s+)?records)",
            r"(?:books\s+and\s+records\s+shall\s+be\s+subject\s+to\s+audit)",
        ]
        return self._presence_regex_match(text, cat, doc_id, patterns, confidence=0.75)

    def _extract_insurance(
        self, text: str, cat: str, doc_id: Optional[str]
    ) -> ExtractionResult:
        patterns = [
            r"(?:maintain\s+(?:commercial\s+general\s+liability\s+)?insurance)",
            r"(?:maintain\s+insurance\s+coverage)",
            r"(?:policies\s+of\s+insurance)",
        ]
        return self._presence_regex_match(text, cat, doc_id, patterns, confidence=0.75)

    def _extract_license_grant(
        self, text: str, cat: str, doc_id: Optional[str]
    ) -> ExtractionResult:
        patterns = [
            r"(?:hereby\s+grants\s+to\s+[\w\s,]+(?:a\s+)?(?:non[\s-]exclusive|exclusive)?\s+license)",
            r"(?:grant\s+of\s+license)",
        ]
        return self._presence_regex_match(text, cat, doc_id, patterns, confidence=0.75)

    def _extract_cap_on_liability(
        self, text: str, cat: str, doc_id: Optional[str]
    ) -> ExtractionResult:
        patterns = [
            r"(?:in\s+no\s+event\s+shall\s+[\w\s\',]+\s+(?:total|aggregate)\s+liability\s+(?:under\s+this\s+agreement\s+)?exceed\s+[\$\d,\w\s]+)",
            r"(?:aggregate\s+liability\s+(?:for\s+all\s+claims\s+)?shall\s+not\s+exceed\s+[\$\d,\w\s]+)",
        ]
        return self._first_regex_match(text, cat, doc_id, patterns, confidence=0.7)

    def _extract_uncapped_liability(
        self, text: str, cat: str, doc_id: Optional[str]
    ) -> ExtractionResult:
        patterns = [
            r"(?:limitation\s+of\s+liability\s+shall\s+not\s+apply\s+to)",
            r"(?:uncapped\s+liability)",
            r"(?:shall\s+not\s+limit\s+either\s+party\'s\s+liability\s+for)",
        ]
        return self._presence_regex_match(text, cat, doc_id, patterns, confidence=0.7)

    def _extract_liquidated_damages(
        self, text: str, cat: str, doc_id: Optional[str]
    ) -> ExtractionResult:
        patterns = [
            r"(?:as\s+liquidated\s+damages\s+and\s+not\s+as\s+a\s+penalty)",
            r"(?:termination\s+fee\s+of\s+[\$\d,\w\s]+)",
            r"(?:liquidated\s+damages)",
        ]
        return self._presence_regex_match(text, cat, doc_id, patterns, confidence=0.7)

    def _extract_renewal_term(
        self, text: str, cat: str, doc_id: Optional[str]
    ) -> ExtractionResult:
        patterns = [
            r"(?:automatically\s+renew(?:ed)?\s+for\s+(?:successive\s+)?(?:terms\s+of\s+)?(\d+\s+(?:year|years|month|months)))",
            r"(?:renewal\s+term\s+of\s+(\d+\s+(?:year|years|month|months)))",
        ]
        return self._first_regex_match(text, cat, doc_id, patterns, confidence=0.75)

    def _extract_notice_period_to_terminate_renewal(
        self, text: str, cat: str, doc_id: Optional[str]
    ) -> ExtractionResult:
        patterns = [
            r"(?:at\s+least\s+(\d+\s+days?|\d+\s+months?)\s+(?:prior\s+)?written\s+notice\s+prior\s+to\s+(?:the\s+end\s+of\s+)?(?:the\s+)?(?:initial\s+term|renewal))",
            r"(?:notice\s+of\s+non[\s-]renewal\s+at\s+least\s+(\d+\s+days?|\d+\s+months?))",
        ]
        return self._first_regex_match(text, cat, doc_id, patterns, confidence=0.75)

    def _extract_expiration_date(
        self, text: str, cat: str, doc_id: Optional[str]
    ) -> ExtractionResult:
        patterns = [
            r"(?:shall\s+expire\s+on\s+)([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
            r"(?:initial\s+term\s+shall\s+be\s+(\d+\s+years?|\d+\s+months?))",
        ]
        return self._first_regex_match(text, cat, doc_id, patterns, confidence=0.75)

    def _extract_parties(
        self, text: str, cat: str, doc_id: Optional[str]
    ) -> ExtractionResult:
        preamble = text[:1500]
        pattern = r"(?:by\s+and\s+between|between)\s+([^,]+?)(?:,\s+a\s+[^,]+)?\s+(?:and|&)\s+([^,]+?)(?:,\s+a\s+[^,]+)?[,;]"
        match = re.search(pattern, preamble, re.IGNORECASE)
        if match:
            p1 = match.group(1).strip()
            p2 = match.group(2).strip()
            val = f"{p1}; {p2}"
            return ExtractionResult(
                category=cat,
                extracted_value=val,
                clause_text=match.group(0),
                evidence=[match.group(0)],
                confidence=0.8,
                status=ExtractionStatus.FOUND,
                source_document_identifier=doc_id,
            )
        return ExtractionResult.not_found(cat, doc_id)

    def _extract_generic(
        self, text: str, cat: str, doc_id: Optional[str]
    ) -> ExtractionResult:
        return ExtractionResult.not_found(cat, doc_id)

    def _first_regex_match(
        self, text: str, cat: str, doc_id: Optional[str], patterns: List[str], confidence: float
    ) -> ExtractionResult:
        for p in patterns:
            match = re.search(p, text, re.IGNORECASE)
            if match:
                val = match.group(1) if match.groups() else match.group(0)
                snippet_start = max(0, match.start() - 50)
                snippet_end = min(len(text), match.end() + 50)
                clause = text[snippet_start:snippet_end].strip()
                return ExtractionResult(
                    category=cat,
                    extracted_value=val.strip(),
                    clause_text=clause,
                    evidence=[match.group(0)],
                    confidence=confidence,
                    status=ExtractionStatus.FOUND,
                    source_document_identifier=doc_id,
                )
        return ExtractionResult.not_found(cat, doc_id)

    def _presence_regex_match(
        self, text: str, cat: str, doc_id: Optional[str], patterns: List[str], confidence: float
    ) -> ExtractionResult:
        for p in patterns:
            match = re.search(p, text, re.IGNORECASE)
            if match:
                snippet_start = max(0, match.start() - 50)
                snippet_end = min(len(text), match.end() + 50)
                clause = text[snippet_start:snippet_end].strip()
                return ExtractionResult(
                    category=cat,
                    extracted_value="Yes",
                    clause_text=clause,
                    evidence=[match.group(0)],
                    confidence=confidence,
                    status=ExtractionStatus.FOUND,
                    source_document_identifier=doc_id,
                )
        return ExtractionResult.not_found(cat, doc_id)

