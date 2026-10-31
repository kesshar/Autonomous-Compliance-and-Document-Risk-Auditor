"""Normalization utilities for contract text, dates, parties, and metrics."""

from __future__ import annotations

import re
import string
from typing import Optional, Set
from dateutil import parser as date_parser


def normalize_text(text: Optional[str]) -> str:
    """Normalizes generic text by lowercasing, stripping punctuation, and collapsing whitespace."""
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Replace slashes and hyphens with spaces for token isolation
    text = text.replace("/", " ").replace("-", " ")
    # Remove punctuation
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    # Collapse multiple whitespaces
    return " ".join(text.split())


def normalize_date(date_str: Optional[str]) -> Optional[str]:
    """Attempts to parse diverse legal date expressions into standard ISO YYYY-MM-DD.

    Examples:
    - '7th day of September, 1999' -> '1999-09-07'
    - 'September 7, 1999' -> '1999-09-07'
    - '10/24/2005' -> '2005-10-24'
    """
    if not date_str:
        return None
    cleaned = date_str.strip()
    # Strip common legal prefixes
    cleaned = re.sub(r"^(as of|dated|entered into on|effective)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)\s+day\s+of\s+", r"\1 ", cleaned, flags=re.IGNORECASE)

    try:
        dt = date_parser.parse(cleaned, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def normalize_boolean(value: Optional[str]) -> Optional[bool]:
    """Normalizes presence/absence strings into booleans."""
    if not value:
        return None
    norm = value.strip().lower()
    if norm in {"yes", "true", "1", "present", "found"}:
        return True
    if norm in {"no", "false", "0", "not present", "not found", "none", "n/a"}:
        return False
    return None


def normalize_party(party: Optional[str]) -> str:
    """Normalizes entity names by stripping legal corporate form suffixes and punctuation."""
    if not party:
        return ""
    text = normalize_text(party)
    suffixes = [
        "a delaware corporation",
        "a nevada corporation",
        "a california corporation",
        "a new york corporation",
        "corporation",
        "incorporated",
        "inc",
        "llc",
        "l l c",
        "limited",
        "ltd",
        "corp",
        "company",
        "co",
    ]
    words = text.split()
    joined = " " + " ".join(words) + " "
    for suffix in suffixes:
        pattern = r"\b" + re.escape(suffix) + r"\b"
        joined = re.sub(pattern, "", joined)
    return " ".join(joined.split())


def compute_token_jaccard(gt: str, pred: str) -> float:
    """Calculates word-level Jaccard similarity index between ground-truth and prediction.

    Follows the official CUAD benchmark evaluation protocol:
    Punctuation is removed, lowercased, and set intersection over union is computed.
    """
    norm_gt = normalize_text(gt)
    norm_pred = normalize_text(pred)

    words_gt: Set[str] = set(norm_gt.split())
    words_pred: Set[str] = set(norm_pred.split())

    if not words_gt and not words_pred:
        return 1.0
    if not words_gt or not words_pred:
        return 0.0

    intersection = words_gt.intersection(words_pred)
    union = words_gt.union(words_pred)
    return len(intersection) / len(union)


def check_exact_match(gt: str, pred: str, category: Optional[str] = None) -> bool:
    """Checks whether prediction matches ground truth after domain-specific normalization."""
    if not gt and not pred:
        return True
    if not gt or not pred:
        return False

    cat_lower = (category or "").lower()

    if "date" in cat_lower:
        d_gt = normalize_date(gt)
        d_pred = normalize_date(pred)
        if d_gt and d_pred:
            return d_gt == d_pred

    if "parties" in cat_lower:
        p_gt = normalize_party(gt)
        p_pred = normalize_party(pred)
        return p_gt == p_pred or (p_gt in p_pred) or (p_pred in p_gt)

    return normalize_text(gt) == normalize_text(pred)

