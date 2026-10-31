"""Pytest fixtures for CUAD dataset integration and contract extraction tests."""

import json
from pathlib import Path
import pytest

MINI_SQUAD_JSON = {
    "version": "aok_v1.0",
    "data": [
        {
            "title": "DISTRIBUTOR_AGREEMENT_ACME_CORP",
            "paragraphs": [
                {
                    "context": (
                        "DISTRIBUTOR AGREEMENT\n"
                        "This Agreement is entered into as of September 7, 1999, by and between "
                        "Acme Corp., a Delaware corporation, and Global Supply LLC, a Nevada corporation.\n"
                        "Section 1. Term and Expiration. The initial term shall expire on September 7, 2004. "
                        "This Agreement shall automatically renew for successive terms of 1 year, "
                        "unless either party gives at least 30 days prior written notice to terminate renewal.\n"
                        "Section 2. Governing Law. This Agreement shall be governed by the laws of the State of New York.\n"
                        "Section 3. Termination for Convenience. Either party may terminate this agreement without cause upon 60 days notice.\n"
                        "Section 4. Cap on Liability. In no event shall either party's aggregate liability exceed $1,000,000.\n"
                        "Section 5. Insurance. Each party shall maintain commercial general liability insurance.\n"
                        "Section 6. Audit Rights. Company shall have the right to audit the books and records of Distributor.\n"
                        "Section 7. Anti-Assignment. Neither party may assign this agreement without prior written consent.\n"
                        "Section 8. License Grant. Company hereby grants to Distributor a non-exclusive license.\n"
                    ),
                    "qas": [
                        {
                            "id": "DISTRIBUTOR_AGREEMENT_ACME_CORP__Parties",
                            "question": "Highlight the parts related to \"Parties\"",
                            "is_impossible": False,
                            "answers": [
                                {"text": "Acme Corp.", "answer_start": 94},
                                {"text": "Global Supply LLC", "answer_start": 139},
                            ],
                        },
                        {
                            "id": "DISTRIBUTOR_AGREEMENT_ACME_CORP__Agreement Date",
                            "question": "Highlight the parts related to \"Agreement Date\"",
                            "is_impossible": False,
                            "answers": [{"text": "September 7, 1999", "answer_start": 57}],
                        },
                        {
                            "id": "DISTRIBUTOR_AGREEMENT_ACME_CORP__Governing Law",
                            "question": "Highlight the parts related to \"Governing Law\"",
                            "is_impossible": False,
                            "answers": [{"text": "State of New York", "answer_start": 472}],
                        },
                        {
                            "id": "DISTRIBUTOR_AGREEMENT_ACME_CORP__Liquidated Damages",
                            "question": "Highlight the parts related to \"Liquidated Damages\"",
                            "is_impossible": True,
                            "answers": [],
                        },
                    ],
                }
            ],
        },
        {
            "title": "CONSULTING_AGREEMENT_BETA_INC",
            "paragraphs": [
                {
                    "context": (
                        "CONSULTING AGREEMENT\n"
                        "Dated as of January 15, 2020 between Beta Inc. and Consultant John Doe.\n"
                        "This agreement shall be effective immediately.\n"
                    ),
                    "qas": [
                        {
                            "id": "CONSULTING_AGREEMENT_BETA_INC__Parties",
                            "question": "Highlight the parts related to \"Parties\"",
                            "is_impossible": False,
                            "answers": [
                                {"text": "Beta Inc.", "answer_start": 52},
                                {"text": "Consultant John Doe", "answer_start": 66},
                            ],
                        },
                        {
                            "id": "CONSULTING_AGREEMENT_BETA_INC__Effective Date",
                            "question": "Highlight the parts related to \"Effective Date\"",
                            "is_impossible": False,
                            "answers": [
                                {"text": "effective immediately", "answer_start": 113}
                            ],
                        },
                    ],
                }
            ],
        },
    ],
}

MALFORMED_SQUAD_JSON = {
    "version": "aok_v1.0",
    "data": [
        {
            "title": "CORRUPT_CONTRACT_ZERO_PARAS",
            "paragraphs": [],
        },
        {
            "title": "CORRUPT_OFFSETS",
            "paragraphs": [
                {
                    "context": "Short text.",
                    "qas": [
                        {
                            "id": "CORRUPT_OFFSETS__Governing Law",
                            "question": "Highlight Governing Law",
                            "is_impossible": False,
                            "answers": [
                                {"text": "Out of bounds text that is way too long", "answer_start": 99999},
                                {"text": "Negative start", "answer_start": -10},
                            ],
                        }
                    ],
                }
            ],
        },
    ],
}


@pytest.fixture
def mini_squad_file(tmp_path: Path) -> Path:
    """Creates a temporary valid CUAD SQuAD JSON file."""
    fpath = tmp_path / "mini_cuad.json"
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(MINI_SQUAD_JSON, f)
    return fpath


@pytest.fixture
def malformed_squad_file(tmp_path: Path) -> Path:
    """Creates a temporary malformed CUAD SQuAD JSON file."""
    fpath = tmp_path / "malformed_cuad.json"
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(MALFORMED_SQUAD_JSON, f)
    return fpath


@pytest.fixture
def mini_csv_file(tmp_path: Path) -> Path:
    """Creates a temporary mini master_clauses.csv file."""
    csv_content = (
        "Filename,Parties,Parties-Answer,Agreement Date,Agreement Date-Answer,Governing Law,Governing Law-Answer\n"
        "CONTRACT_A,\"Acme and Beta\",\"Acme; Beta\",\"Dated 2021-01-01\",\"2021-01-01\",\"Governed by Delaware\",\"Delaware\"\n"
        "CONTRACT_B,\"Gamma Corp\",\"Gamma Corp\",\"\",\"\",\"\",\"\"\n"
    )
    fpath = tmp_path / "mini_master_clauses.csv"
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(csv_content)
    return fpath

