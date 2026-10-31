"""Reusable, robust loader for the CUAD v1 dataset (SQuAD JSON and CSV formats)."""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import yaml

from src.data.schema import ClauseAnnotation, ContractDocument, EvidenceSpan

logger = logging.getLogger(__name__)


def load_category_aliases(config_path: Union[str, Path] = "configs/cuad_categories.yaml") -> Dict[str, str]:
    """Loads category configurations and builds an alias-to-canonical-name lookup dictionary."""
    config_file = Path(config_path)
    if not config_file.is_file():
        logger.warning("Category config %s not found. Using standard defaults.", config_file)
        return {}

    with open(config_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    alias_map: Dict[str, str] = {}
    for cat in data.get("categories", []):
        canonical = cat["name"]
        alias_map[canonical.strip().lower()] = canonical
        for alias in cat.get("aliases", []):
            alias_map[alias.strip().lower()] = canonical
    return alias_map


class CUADLoader:
    """Loader for the Contract Understanding Atticus Dataset (CUAD v1).

    Parses official SQuAD 2.0 format JSON and/or master_clauses.csv tabular annotations.
    Features:
    - Normalizes category names and aliases.
    - Preserves multiple evidence spans per clause.
    - Gracefully handles missing annotations and malformed records.
    - Prevents data leakage by separating raw contract text from gold annotations.
    """

    def __init__(
        self,
        target_categories: Optional[List[str]] = None,
        config_path: Union[str, Path] = "configs/cuad_categories.yaml",
    ):
        self.alias_map = load_category_aliases(config_path)
        self.target_categories: Optional[Set[str]] = None
        if target_categories is not None:
            self.target_categories = {
                self.resolve_category_name(c) for c in target_categories
            }

    def resolve_category_name(self, raw_category: str) -> str:
        """Resolves a raw or aliased category name to its canonical CUAD label."""
        cleaned = raw_category.strip()
        cleaned_lower = cleaned.lower()
        return self.alias_map.get(cleaned_lower, cleaned)

    def is_category_selected(self, category_name: str) -> bool:
        """Checks if a category is among the selected targets."""
        if self.target_categories is None:
            return True
        canonical = self.resolve_category_name(category_name)
        return canonical in self.target_categories

    def parse_squad_json(
        self,
        json_path: Union[str, Path],
        max_documents: Optional[int] = None,
    ) -> Dict[str, ContractDocument]:
        """Parses a SQuAD 2.0-format CUAD JSON file (e.g., CUAD_v1.json or test.json).

        Returns a dictionary mapping contract_id to ContractDocument.
        """
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"CUAD JSON file not found at: {path}")

        logger.info("Parsing CUAD JSON file: %s", path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            try:
                payload = json.load(f)
            except json.JSONDecodeError as err:
                raise ValueError(f"Malformed JSON in CUAD file {path}: {err}") from err

        raw_contracts = payload.get("data", [])
        documents: Dict[str, ContractDocument] = {}

        for item_idx, contract_data in enumerate(raw_contracts):
            if max_documents is not None and len(documents) >= max_documents:
                break

            title = contract_data.get("title", f"contract_{item_idx}").strip()
            paragraphs = contract_data.get("paragraphs", [])
            if not paragraphs:
                logger.warning("Contract %s has no paragraphs; skipping.", title)
                continue

            # In CUAD, each contract is stored in a single paragraph with full context
            context_text = paragraphs[0].get("context", "")
            qas = paragraphs[0].get("qas", [])

            annotations: Dict[str, ClauseAnnotation] = {}

            for qa in qas:
                qa_id = qa.get("id", "")
                question_text = qa.get("question", "")

                # Extract category name from qa_id or question
                raw_cat_name = self._extract_category_from_qa(qa_id, question_text)
                if not raw_cat_name:
                    continue

                canonical_cat = self.resolve_category_name(raw_cat_name)
                if not self.is_category_selected(canonical_cat):
                    continue

                is_impossible = qa.get("is_impossible", False)
                raw_answers = qa.get("answers", [])

                evidence_spans: List[EvidenceSpan] = []
                answers: List[str] = []

                if not is_impossible and raw_answers:
                    for ans in raw_answers:
                        ans_text = ans.get("text", "")
                        ans_start = ans.get("answer_start", -1)

                        if not ans_text or ans_start < 0:
                            continue

                        ans_end = ans_start + len(ans_text)

                        # Validate offset boundaries against context
                        if ans_end <= len(context_text):
                            extracted_sub = context_text[ans_start:ans_end]
                            # Soft match check in case of whitespace anomalies
                            if extracted_sub != ans_text:
                                logger.debug(
                                    "Span text offset minor mismatch in %s for %s",
                                    title,
                                    canonical_cat,
                                )

                        evidence_spans.append(
                            EvidenceSpan(
                                text=ans_text,
                                start_char=ans_start,
                                end_char=ans_end,
                            )
                        )
                        answers.append(ans_text)

                is_present = len(answers) > 0 and not is_impossible

                # Handle duplicate QA entries for the same category by merging spans
                if canonical_cat in annotations:
                    existing = annotations[canonical_cat]
                    merged_answers = list(dict.fromkeys(existing.answers + answers))
                    merged_spans = existing.evidence_spans + [
                        s for s in evidence_spans if s not in existing.evidence_spans
                    ]
                    annotations[canonical_cat] = ClauseAnnotation(
                        contract_id=title,
                        category_name=canonical_cat,
                        is_present=existing.is_present or is_present,
                        answers=merged_answers,
                        evidence_spans=merged_spans,
                        raw_question=existing.raw_question or question_text,
                    )
                else:
                    annotations[canonical_cat] = ClauseAnnotation(
                        contract_id=title,
                        category_name=canonical_cat,
                        is_present=is_present,
                        answers=answers,
                        evidence_spans=evidence_spans,
                        raw_question=question_text,
                    )

            doc = ContractDocument(
                id=title,
                title=title,
                text=context_text,
                annotations=annotations,
                metadata={"source_file": str(path), "total_annotations": len(annotations)},
            )
            documents[title] = doc

        logger.info("Successfully loaded %d documents from %s", len(documents), path)
        return documents

    def parse_master_clauses_csv(
        self,
        csv_path: Union[str, Path],
        contracts_dir: Optional[Union[str, Path]] = None,
        max_documents: Optional[int] = None,
    ) -> Dict[str, ContractDocument]:
        """Parses CUAD master_clauses.csv containing tabular clauses and answers.

        Optionally links to contract text files in contracts_dir (e.g. full_contracts_txt/).
        """
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"CUAD master_clauses.csv file not found at: {path}")

        logger.info("Parsing CUAD CSV file: %s", path)
        contracts_path = Path(contracts_dir) if contracts_dir else None

        documents: Dict[str, ContractDocument] = {}

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            # Identify category columns. Each category has "<Category>" and "<Category>-Answer"
            answer_cols = [c for c in fieldnames if c.endswith("-Answer")]
            cat_pairs: List[Tuple[str, str, str]] = []
            for ans_col in answer_cols:
                base_cat = ans_col[:-7]
                if base_cat in fieldnames:
                    cat_pairs.append((base_cat, ans_col, self.resolve_category_name(base_cat)))

            for row_idx, row in enumerate(reader):
                if max_documents is not None and len(documents) >= max_documents:
                    break

                filename = row.get("Filename", f"contract_{row_idx}").strip()
                if not filename:
                    continue

                # Load full contract text from txt file if available
                doc_text = ""
                if contracts_path and contracts_path.is_dir():
                    txt_candidate = contracts_path / f"{filename}.txt"
                    if not txt_candidate.is_file():
                        # Try without extension or search case-insensitively
                        candidates = list(contracts_path.glob(f"*{filename}*.txt"))
                        if candidates:
                            txt_candidate = candidates[0]
                    if txt_candidate.is_file():
                        try:
                            with open(txt_candidate, "r", encoding="utf-8", errors="replace") as tf:
                                doc_text = tf.read()
                        except Exception as e:
                            logger.debug("Failed reading txt for %s: %e", filename, e)

                annotations: Dict[str, ClauseAnnotation] = {}

                for base_col, ans_col, canonical_cat in cat_pairs:
                    if not self.is_category_selected(canonical_cat):
                        continue

                    context_clause = (row.get(base_col) or "").strip()
                    answer_text = (row.get(ans_col) or "").strip()

                    is_present = bool(answer_text or context_clause)
                    answers = [answer_text] if answer_text else []
                    spans: List[EvidenceSpan] = []

                    if context_clause:
                        # Find start/end offset in doc_text if available
                        start_char = 0
                        end_char = len(context_clause)
                        if doc_text and context_clause in doc_text:
                            start_char = doc_text.index(context_clause)
                            end_char = start_char + len(context_clause)

                        spans.append(
                            EvidenceSpan(
                                text=context_clause,
                                start_char=start_char,
                                end_char=end_char,
                            )
                        )

                    annotations[canonical_cat] = ClauseAnnotation(
                        contract_id=filename,
                        category_name=canonical_cat,
                        is_present=is_present,
                        answers=answers,
                        evidence_spans=spans,
                    )

                doc = ContractDocument(
                    id=filename,
                    title=filename,
                    text=doc_text or " ".join(s.text for a in annotations.values() for s in a.evidence_spans),
                    annotations=annotations,
                    metadata={"source_file": str(path), "total_annotations": len(annotations)},
                )
                documents[filename] = doc

        logger.info("Successfully loaded %d documents from %s", len(documents), path)
        return documents

    def _extract_category_from_qa(self, qa_id: str, question: str) -> Optional[str]:
        """Extracts the category name from SQuAD QA id (e.g., 'DOC_TITLE__Category Name')

        or from the question prompt format:
        'Highlight the parts (if any) of this contract related to "<Category>" ...'
        """
        if "__" in qa_id:
            parts = qa_id.split("__", 1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip()

        # Regex fallback on question text
        match = re.search(r'related to "([^"]+)"', question)
        if match:
            return match.group(1).strip()

        return None

