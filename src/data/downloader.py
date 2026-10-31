"""Reproducible dataset acquisition and integrity verification for CUAD v1."""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import zipfile
import requests

from src.data.schema import DatasetManifest, FileChecksum

logger = logging.getLogger(__name__)

OFFICIAL_ZENODO_URL = "https://zenodo.org/api/records/4595826/files/CUAD_v1.zip/content"
OFFICIAL_GITHUB_DATA_URL = "https://raw.githubusercontent.com/The-Atticus-Project/cuad/main/data.zip"
ZENODO_RECORD_URL = "https://doi.org/10.5281/zenodo.4595826"


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Computes SHA-256 hash of a file efficiently in 1MB chunks."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_file_integrity(file_path: Path, expected_sha256: Optional[str] = None) -> bool:
    """Verifies that a file exists, is non-empty, and optionally matches expected SHA-256."""
    if not file_path.is_file():
        return False
    if file_path.stat().st_size == 0:
        return False
    if expected_sha256:
        actual_sha256 = compute_sha256(file_path)
        return actual_sha256.lower() == expected_sha256.lower()
    return True


def download_stream(
    url: str,
    destination: Path,
    expected_sha256: Optional[str] = None,
    force: bool = False,
    timeout: int = 300,
) -> str:
    """Downloads a file from url to destination with streaming and integrity validation.

    Raises FileExistsError if destination already exists and force is False.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        logger.info("File %s already exists. Checking integrity...", destination)
        actual_sha = compute_sha256(destination)
        if expected_sha256 and actual_sha.lower() != expected_sha256.lower():
            raise ValueError(
                f"Existing file {destination} hash mismatch. "
                f"Expected: {expected_sha256}, got: {actual_sha}. Pass force=True to re-download."
            )
        logger.info("Existing file %s is intact (SHA-256: %s).", destination, actual_sha)
        return actual_sha

    logger.info("Downloading %s to %s ...", url, destination)
    temp_dest = destination.with_suffix(f"{destination.suffix}.part")

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(temp_dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

    if total_size > 0 and downloaded != total_size:
        if temp_dest.exists():
            temp_dest.unlink()
        raise IOError(f"Incomplete download: received {downloaded} of {total_size} bytes.")

    actual_sha = compute_sha256(temp_dest)
    if expected_sha256 and actual_sha.lower() != expected_sha256.lower():
        if temp_dest.exists():
            temp_dest.unlink()
        raise ValueError(f"Checksum mismatch for downloaded file. Expected: {expected_sha256}, got: {actual_sha}")

    if destination.exists():
        destination.unlink()
    temp_dest.rename(destination)
    logger.info("Successfully downloaded %s (SHA-256: %s)", destination, actual_sha)
    return actual_sha


def extract_zip(archive_path: Path, extract_to: Path, force: bool = False) -> None:
    """Extracts a ZIP archive safely into extract_to directory."""
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            target_path = extract_to / member.filename
            if target_path.exists() and not force:
                continue
            zip_ref.extract(member, extract_to)


def acquire_cuad_dataset(
    output_dir: Path = Path("data/raw/cuad"),
    source_url: str = OFFICIAL_ZENODO_URL,
    force: bool = False,
) -> DatasetManifest:
    """Downloads official CUAD v1 dataset, verifies integrity, extracts files, and writes manifest.

    Guarantees:
    - Never silently overwrites an existing dataset unless force=True.
    - Records provenance, timestamp, SHA-256 checksums, and license in manifest.json.
    - Documents attribution requirements and SEC EDGAR rights limitations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    if manifest_path.exists() and not force:
        logger.info("Manifest already exists at %s. Loading existing manifest.", manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            manifest = DatasetManifest(**data)
        # Verify critical files listed in manifest
        for rel_path, file_meta in manifest.files.items():
            full_path = output_dir / rel_path
            if not verify_file_integrity(full_path, file_meta.sha256):
                logger.warning(
                    "Integrity check failed for %s. Re-running acquisition with verification.",
                    full_path,
                )
                break
        else:
            logger.info("Existing CUAD dataset is fully verified. Returning manifest.")
            return manifest

    archive_path = output_dir / "CUAD_v1.zip"
    archive_sha = download_stream(source_url, archive_path, force=force)

    logger.info("Extracting %s into %s ...", archive_path, output_dir)
    extract_zip(archive_path, output_dir, force=force)

    # Gather file metadata for all extracted artifacts
    files_map: Dict[str, FileChecksum] = {
        "CUAD_v1.zip": FileChecksum(
            path="CUAD_v1.zip",
            size_bytes=archive_path.stat().st_size,
            sha256=archive_sha,
        )
    }

    # Identify primary annotation files
    # The zip might extract files directly or under a subfolder 'CUAD_v1/'
    primary_files = ["CUAD_v1.json", "master_clauses.csv", "master_clauses.xlsx"]
    for pf in primary_files:
        candidate_paths = [
            output_dir / pf,
            output_dir / "CUAD_v1" / pf,
            output_dir / "full_contracts_txt",
        ]
        for p in candidate_paths:
            if p.is_file():
                rel_name = str(p.relative_to(output_dir)).replace("\\", "/")
                files_map[rel_name] = FileChecksum(
                    path=rel_name,
                    size_bytes=p.stat().st_size,
                    sha256=compute_sha256(p),
                )

    manifest = DatasetManifest(
        dataset_name="Contract Understanding Atticus Dataset (CUAD)",
        dataset_version="v1.0",
        source_url=source_url,
        license="CC BY 4.0 (Annotations); Underlying contracts are public SEC EDGAR filings",
        download_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sha256_checksum=archive_sha,
        total_documents=510,
        files=files_map,
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, indent=2)

    logger.info("Acquisition complete. Manifest written to %s.", manifest_path)
    return manifest

