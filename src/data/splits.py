"""Reproducible, deterministic document-level dataset splitting.

Guarantees:
- Contracts are never split across train/validation/test (no clause leakage).
- Deterministic and reproducible across operating systems via sorted IDs and fixed seed.
- Persists split manifests for auditability and verification.
"""

from __future__ import annotations

import datetime
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field

from src.data.schema import ContractDocument

logger = logging.getLogger(__name__)


class SplitConfig(BaseModel):
    """Configuration for document-level dataset partitioning."""

    train_ratio: float = Field(0.70, ge=0.0, le=1.0)
    val_ratio: float = Field(0.10, ge=0.0, le=1.0)
    test_ratio: float = Field(0.20, ge=0.0, le=1.0)
    random_seed: int = 42

    def validate_ratios(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Split ratios must sum to 1.0, got: {total}")


class SplitManifest(BaseModel):
    """Manifest recording document IDs per split and partition metadata."""

    created_at: str
    random_seed: int
    strategy: str
    counts: Dict[str, int]
    splits: Dict[str, List[str]]


def create_document_splits(
    document_ids: List[str],
    config: Optional[SplitConfig] = None,
    official_test_ids: Optional[List[str]] = None,
) -> SplitManifest:
    """Creates deterministic document-level splits with zero clause leakage.

    Args:
        document_ids: List of unique contract IDs.
        config: Ratios and random seed configuration.
        official_test_ids: Optional list of contract IDs designated as official test set.
    """
    if config is None:
        config = SplitConfig()
    config.validate_ratios()

    # Deduplicate and sort IDs to guarantee cross-platform deterministic order
    unique_ids = sorted(list(set(document_ids)))
    total = len(unique_ids)

    if total == 0:
        raise ValueError("Cannot create splits from an empty document list.")

    test_ids: List[str] = []
    val_ids: List[str] = []
    train_ids: List[str] = []

    if official_test_ids:
        official_set = set(official_test_ids)
        test_ids = [d for d in unique_ids if d in official_set]
        remaining = [d for d in unique_ids if d not in official_set]
        strategy = "official_test_with_deterministic_train_val"

        # Split remaining into train and val based on relative ratio
        rng = random.Random(config.random_seed)
        rng.shuffle(remaining)

        val_share = config.val_ratio / (config.train_ratio + config.val_ratio)
        n_val = max(1, int(round(len(remaining) * val_share))) if remaining and config.val_ratio > 0 else 0
        val_ids = sorted(remaining[:n_val])
        train_ids = sorted(remaining[n_val:])
        test_ids = sorted(test_ids)
    else:
        strategy = f"deterministic_seeded_shuffle (seed={config.random_seed})"
        shuffled = list(unique_ids)
        rng = random.Random(config.random_seed)
        rng.shuffle(shuffled)

        n_test = max(1, int(round(total * config.test_ratio))) if config.test_ratio > 0 else 0
        n_val = max(1, int(round(total * config.val_ratio))) if config.val_ratio > 0 else 0

        # Ensure no negative remainder
        if n_test + n_val >= total:
            n_test = max(1, total // 5)
            n_val = max(1, total // 10)

        test_ids = sorted(shuffled[:n_test])
        val_ids = sorted(shuffled[n_test : n_test + n_val])
        train_ids = sorted(shuffled[n_test + n_val :])

    # Strict Invariant Verification
    set_train = set(train_ids)
    set_val = set(val_ids)
    set_test = set(test_ids)

    assert set_train.isdisjoint(set_val), "Train and validation splits overlap!"
    assert set_train.isdisjoint(set_test), "Train and test splits overlap!"
    assert set_val.isdisjoint(set_test), "Validation and test splits overlap!"
    assert (set_train | set_val | set_test) == set(unique_ids), "Some documents were dropped during splitting!"

    manifest = SplitManifest(
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        random_seed=config.random_seed,
        strategy=strategy,
        counts={
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "total": total,
        },
        splits={
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        },
    )

    logger.info(
        "Splits created successfully: Train=%d, Val=%d, Test=%d (Total=%d)",
        len(train_ids),
        len(val_ids),
        len(test_ids),
        total,
    )
    return manifest


def save_splits(manifest: SplitManifest, output_path: Union[str, Path]) -> None:
    """Saves the split manifest to disk as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, indent=2)
    logger.info("Saved split manifest to: %s", path)


def load_splits(manifest_path: Union[str, Path]) -> SplitManifest:
    """Loads a previously saved split manifest."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SplitManifest(**data)


def filter_documents_by_split(
    documents: Dict[str, ContractDocument],
    manifest: SplitManifest,
    split_name: str,
) -> Dict[str, ContractDocument]:
    """Filters a dictionary of ContractDocuments to include only those in the specified split."""
    if split_name not in manifest.splits:
        raise KeyError(f"Unknown split '{split_name}'. Available: {list(manifest.splits.keys())}")
    target_ids = set(manifest.splits[split_name])
    return {doc_id: doc for doc_id, doc in documents.items() if doc_id in target_ids}


if __name__ == "__main__":
    import argparse
    from src.data.loader import CUADLoader

    parser = argparse.ArgumentParser(description="Generate deterministic document-level splits for CUAD.")
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/cuad/CUAD_v1.json"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/splits.json"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    loader = CUADLoader()
    if args.data_path.suffix.lower() == ".json":
        docs = loader.parse_squad_json(args.data_path)
    else:
        docs = loader.parse_master_clauses_csv(args.data_path)

    cfg = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )
    manifest = create_document_splits(list(docs.keys()), config=cfg)
    save_splits(manifest, args.output_path)
    print(f"Generated splits for {len(docs)} documents -> {args.output_path}")

