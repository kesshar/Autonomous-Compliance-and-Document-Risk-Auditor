"""Unit tests for dataset download, manifest generation, and integrity verification."""

import json
from pathlib import Path
import pytest
from src.data.downloader import (
    compute_sha256,
    verify_file_integrity,
    download_stream,
)
from src.data.schema import DatasetManifest, FileChecksum


def test_compute_sha256(tmp_path: Path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Hello CUAD Dataset", encoding="utf-8")

    sha = compute_sha256(test_file)
    assert len(sha) == 64
    assert isinstance(sha, str)
    # Deterministic hash for "Hello CUAD Dataset"
    assert sha == compute_sha256(test_file)


def test_verify_file_integrity(tmp_path: Path):
    test_file = tmp_path / "valid.txt"
    test_file.write_text("integrity test", encoding="utf-8")
    actual_sha = compute_sha256(test_file)

    assert verify_file_integrity(test_file) is True
    assert verify_file_integrity(test_file, expected_sha256=actual_sha) is True
    assert verify_file_integrity(test_file, expected_sha256="wrong_hash") is False

    non_existent = tmp_path / "ghost.txt"
    assert verify_file_integrity(non_existent) is False


def test_dataset_manifest_schema():
    manifest = DatasetManifest(
        source_url="https://doi.org/10.5281/zenodo.4595826",
        download_timestamp="2026-11-01T00:00:00Z",
        sha256_checksum="abc123456789",
        total_documents=510,
        files={
            "CUAD_v1.zip": FileChecksum(path="CUAD_v1.zip", size_bytes=105883672, sha256="abc123456789")
        },
    )

    assert manifest.dataset_name == "Contract Understanding Atticus Dataset (CUAD)"
    assert "CC BY 4.0" in manifest.license
    assert "The Atticus Project" in manifest.attribution
    assert "SEC EDGAR" in manifest.rights_limitation
    assert manifest.total_documents == 510

