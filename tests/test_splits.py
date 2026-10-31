"""Unit tests for deterministic document-level dataset splitting."""

from pathlib import Path
import pytest
from src.data.splits import (
    SplitConfig,
    create_document_splits,
    save_splits,
    load_splits,
    filter_documents_by_split,
)
from src.data.schema import ContractDocument


def test_deterministic_document_splits():
    doc_ids = [f"CONTRACT_{i:03d}" for i in range(100)]
    config = SplitConfig(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_seed=42)

    manifest_1 = create_document_splits(doc_ids, config=config)
    manifest_2 = create_document_splits(doc_ids, config=config)

    # Determinism across identical seeds
    assert manifest_1.splits["train"] == manifest_2.splits["train"]
    assert manifest_1.splits["val"] == manifest_2.splits["val"]
    assert manifest_1.splits["test"] == manifest_2.splits["test"]

    # Exact counts
    assert manifest_1.counts["train"] == 70
    assert manifest_1.counts["val"] == 10
    assert manifest_1.counts["test"] == 20

    # Strict mutual exclusivity (no document overlap)
    set_train = set(manifest_1.splits["train"])
    set_val = set(manifest_1.splits["val"])
    set_test = set(manifest_1.splits["test"])

    assert len(set_train.intersection(set_val)) == 0
    assert len(set_train.intersection(set_test)) == 0
    assert len(set_val.intersection(set_test)) == 0

    # Exhaustive coverage
    assert (set_train | set_val | set_test) == set(doc_ids)


def test_official_test_split_integration():
    doc_ids = [f"CONTRACT_{i:03d}" for i in range(50)]
    official_test = ["CONTRACT_005", "CONTRACT_010", "CONTRACT_015"]

    manifest = create_document_splits(doc_ids, official_test_ids=official_test)
    assert set(manifest.splits["test"]) == set(official_test)
    assert "CONTRACT_005" not in manifest.splits["train"]
    assert "CONTRACT_005" not in manifest.splits["val"]


def test_save_and_load_splits(tmp_path: Path):
    doc_ids = [f"DOC_{i}" for i in range(20)]
    manifest = create_document_splits(doc_ids)

    out_file = tmp_path / "splits.json"
    save_splits(manifest, out_file)
    assert out_file.is_file()

    loaded = load_splits(out_file)
    assert loaded.random_seed == manifest.random_seed
    assert loaded.splits == manifest.splits
    assert loaded.counts == manifest.counts


def test_filter_documents_by_split():
    manifest = create_document_splits(["DOC_A", "DOC_B", "DOC_C", "DOC_D"])
    fake_docs = {
        doc_id: ContractDocument(id=doc_id, title=doc_id, text=f"Text of {doc_id}")
        for doc_id in ["DOC_A", "DOC_B", "DOC_C", "DOC_D"]
    }

    test_docs = filter_documents_by_split(fake_docs, manifest, "test")
    assert all(doc_id in manifest.splits["test"] for doc_id in test_docs.keys())


def test_invalid_split_ratios():
    config = SplitConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        config.validate_ratios()

