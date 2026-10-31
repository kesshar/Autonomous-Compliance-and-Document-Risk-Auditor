"""Unit tests for contract text, date, entity, and metric normalizers."""

from src.extraction.normalizer import (
    check_exact_match,
    compute_token_jaccard,
    normalize_boolean,
    normalize_date,
    normalize_party,
    normalize_text,
)


def test_normalize_text():
    raw = "  This, Agreement; (dated 1999) - is Valid. \n\t "
    expected = "this agreement dated 1999 is valid"
    assert normalize_text(raw) == expected


def test_normalize_date():
    assert normalize_date("7th day of September, 1999") == "1999-09-07"
    assert normalize_date("September 7, 1999") == "1999-09-07"
    assert normalize_date("dated as of October 24, 2015") == "2015-10-24"
    assert normalize_date("2020-05-01") == "2020-05-01"
    assert normalize_date("not a date") is None
    assert normalize_date("") is None


def test_normalize_boolean():
    assert normalize_boolean("Yes") is True
    assert normalize_boolean("TRUE") is True
    assert normalize_boolean("present") is True
    assert normalize_boolean("No") is False
    assert normalize_boolean("NOT FOUND") is False
    assert normalize_boolean("gibberish") is None


def test_normalize_party():
    p1 = "Acme Corp., a Delaware corporation"
    p2 = "Acme Corporation"
    p3 = "Acme Inc."
    p4 = "Acme"

    assert normalize_party(p1) == "acme"
    assert normalize_party(p2) == "acme"
    assert normalize_party(p3) == "acme"
    assert normalize_party(p4) == "acme"


def test_compute_token_jaccard():
    t1 = "State of New York"
    t2 = "the laws of the State of New York"
    jaccard = compute_token_jaccard(t1, t2)
    assert 0.5 <= jaccard <= 1.0

    t3 = "State of California"
    assert compute_token_jaccard(t1, t3) < 0.5

    assert compute_token_jaccard("", "") == 1.0
    assert compute_token_jaccard("apple", "") == 0.0


def test_check_exact_match():
    # Date exact match
    assert check_exact_match("September 7, 1999", "1999-09-07", category="Agreement Date") is True

    # Party exact match
    assert check_exact_match("Acme Corp., a Delaware corporation", "Acme Inc.", category="Parties") is True

    # Generic text exact match
    assert check_exact_match("State of New York", "state of new york") is True
    assert check_exact_match("Delaware", "New York") is False

