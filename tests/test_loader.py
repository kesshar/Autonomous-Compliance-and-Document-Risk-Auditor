"""Unit tests for CUAD dataset loader."""

from pathlib import Path
import pytest
from src.data.loader import CUADLoader
from src.data.schema import ContractDocument


def test_parse_squad_json_success(mini_squad_file: Path):
    loader = CUADLoader()
    docs = loader.parse_squad_json(mini_squad_file)

    assert len(docs) == 2
    assert "DISTRIBUTOR_AGREEMENT_ACME_CORP" in docs
    doc = docs["DISTRIBUTOR_AGREEMENT_ACME_CORP"]

    assert isinstance(doc, ContractDocument)
    assert len(doc.text) > 100
    assert "Parties" in doc.annotations

    parties_anno = doc.annotations["Parties"]
    assert parties_anno.is_present is True
    assert len(parties_anno.answers) == 2
    assert "Acme Corp." in parties_anno.answers
    assert "Global Supply LLC" in parties_anno.answers
    assert len(parties_anno.evidence_spans) == 2
    assert parties_anno.evidence_spans[0].text == "Acme Corp."
    assert parties_anno.evidence_spans[0].start_char == 94


def test_missing_annotations_handling(mini_squad_file: Path):
    loader = CUADLoader()
    docs = loader.parse_squad_json(mini_squad_file)
    doc = docs["DISTRIBUTOR_AGREEMENT_ACME_CORP"]

    # Liquidated Damages was marked is_impossible=True with empty answers
    assert "Liquidated Damages" in doc.annotations
    ld_anno = doc.annotations["Liquidated Damages"]
    assert ld_anno.is_present is False
    assert len(ld_anno.answers) == 0
    assert len(ld_anno.evidence_spans) == 0


def test_unlabeled_context_prevents_leakage(mini_squad_file: Path):
    loader = CUADLoader()
    docs = loader.parse_squad_json(mini_squad_file)
    doc = docs["DISTRIBUTOR_AGREEMENT_ACME_CORP"]

    unlabeled = doc.get_unlabeled_context()
    assert "id" in unlabeled
    assert "title" in unlabeled
    assert "text" in unlabeled
    assert "annotations" not in unlabeled
    assert "answers" not in unlabeled
    assert "evidence_spans" not in unlabeled


def test_parse_master_clauses_csv_success(mini_csv_file: Path):
    loader = CUADLoader()
    docs = loader.parse_master_clauses_csv(mini_csv_file)

    assert len(docs) == 2
    assert "CONTRACT_A" in docs
    doc_a = docs["CONTRACT_A"]
    assert "Governing Law" in doc_a.annotations
    assert doc_a.annotations["Governing Law"].is_present is True
    assert "Delaware" in doc_a.annotations["Governing Law"].answers

    doc_b = docs["CONTRACT_B"]
    assert doc_b.annotations["Agreement Date"].is_present is False
    assert len(doc_b.annotations["Agreement Date"].answers) == 0


def test_malformed_records_handling(malformed_squad_file: Path):
    loader = CUADLoader()
    docs = loader.parse_squad_json(malformed_squad_file)

    # Contract with zero paragraphs should be safely skipped
    assert "CORRUPT_CONTRACT_ZERO_PARAS" not in docs
    # Contract with negative or out-of-bounds start offsets should parse without crashing
    assert "CORRUPT_OFFSETS" in docs
    doc = docs["CORRUPT_OFFSETS"]
    # Negative start offset answers are filtered out
    anno = doc.annotations.get("Governing Law")
    assert anno is not None
    # Out of bounds text was captured safely
    assert len(anno.evidence_spans) >= 1


def test_category_filtering(mini_squad_file: Path):
    # Only load 'Parties' and 'Agreement Date'
    loader = CUADLoader(target_categories=["Parties", "Agreement Date"])
    docs = loader.parse_squad_json(mini_squad_file)
    doc = docs["DISTRIBUTOR_AGREEMENT_ACME_CORP"]

    assert "Parties" in doc.annotations
    assert "Agreement Date" in doc.annotations
    assert "Governing Law" not in doc.annotations
    assert "Liquidated Damages" not in doc.annotations

