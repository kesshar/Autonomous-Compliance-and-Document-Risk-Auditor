"""Unit tests for CUAD 15 selected contractual attributes and category configuration."""

from pathlib import Path
import yaml
import pytest
from src.data.loader import CUADLoader, load_category_aliases

EXPECTED_15_CATEGORIES = [
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice Period to Terminate Renewal",
    "Governing Law",
    "Termination for Convenience",
    "Cap on Liability",
    "Insurance",
    "Audit Rights",
    "Uncapped Liability",
    "Liquidated Damages",
    "Anti-Assignment",
    "License Grant",
]


def test_categories_yaml_integrity():
    config_path = Path("configs/cuad_categories.yaml")
    assert config_path.is_file(), "cuad_categories.yaml must exist in configs/"

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "categories" in data
    categories = data["categories"]
    assert len(categories) == 15, f"Expected 15 categories, found {len(categories)}"

    names = [c["name"] for c in categories]
    for expected_name in EXPECTED_15_CATEGORIES:
        assert expected_name in names, f"Missing category: {expected_name}"

    for cat in categories:
        assert "name" in cat and cat["name"].strip()
        assert "description" in cat and len(cat["description"]) > 10
        assert "output_type" in cat and cat["output_type"].strip()
        assert "extraction_instructions" in cat and len(cat["extraction_instructions"]) > 10
        assert "task_type" in cat
        assert cat["task_type"] in {"value_extraction", "presence_absence", "presence_or_value"}


def test_alias_resolution():
    alias_map = load_category_aliases("configs/cuad_categories.yaml")
    assert len(alias_map) >= 15

    loader = CUADLoader(config_path="configs/cuad_categories.yaml")

    # Prompt asked for "Notice to Terminate Renewal", CUAD canonical is "Notice Period to Terminate Renewal"
    assert loader.resolve_category_name("Notice to Terminate Renewal") == "Notice Period to Terminate Renewal"
    assert loader.resolve_category_name("Notice Period to Terminate Renewal") == "Notice Period to Terminate Renewal"
    assert loader.resolve_category_name("parties") == "Parties"
    assert loader.resolve_category_name("Governing Law") == "Governing Law"


def test_no_invented_cuad_labels():
    with open("configs/cuad_categories.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = {c["name"] for c in data["categories"]}
    official_cuad_41_subset = set(EXPECTED_15_CATEGORIES)
    assert names == official_cuad_41_subset, "Found invented or unauthorized labels in configuration."

