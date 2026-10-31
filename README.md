# Autonomous Compliance & Document Risk Auditor

An enterprise-grade AI-powered Legal Document Compliance & Risk Auditor. The system analyzes commercial legal agreements, extracts critical contractual attributes, identifies compliance gaps, detects operational risks and contradictions, and generates evidence-backed audit reports.

---

## Milestone 1: CUAD Dataset Integration & Contract Extraction Evaluation

This milestone establishes the dataset ingestion and evaluation infrastructure using the official **Contract Understanding Atticus Dataset (CUAD v1)** created by The Atticus Project.

---

## 1. What is CUAD and Why Are We Using It?

### What is CUAD?
**CUAD (Contract Understanding Atticus Dataset)** is a corpus of 510 commercial legal agreements manually annotated by legal experts with more than 13,000 annotations across 41 clause types. The dataset is formatted for question-answering and span-selection benchmarks (using the SQuAD 2.0 format as well as tabular matrices).

- **Academic Paper**: Hendrycks et al., *"CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review"*, NeurIPS 2021 ([arXiv:2103.06268](https://arxiv.org/abs/2103.06268)).
- **Official Documentation**: [The Atticus Project - CUAD](https://www.atticusprojectai.org/cuad)
- **Dataset Archive**: [Zenodo Record 4595826](https://doi.org/10.5281/zenodo.4595826)

### Why CUAD?
1. **Expert Ground Truth**: Curated and verified by experienced corporate attorneys, providing a gold standard that synthetic data cannot replicate.
2. **Real-World Complexity**: Contracts average over 10,000 words each, featuring complex legal syntax, cross-references, schedules, and exhibits.
3. **Presence and Absence**: Contains both positive clause instances and negative/absent instances (`is_impossible: true`), critical for evaluating false-positive rates in legal risk auditing.
4. **Standardized Benchmarking**: Enables reproducible precision, recall, F1, exact match, and Jaccard IoU evaluations before deploying extraction pipelines to production.

---

## 2. Dataset Attribution & Underlying Rights Limitation

### Attribution Requirement
CUAD annotations are released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.
If you use this dataset in research or development, cite:
```bibtex
@article{hendrycks2021cuad,
  title={CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review},
  author={Dan Hendrycks and Collin Burns and Anya Chen and Spencer Ball},
  journal={NeurIPS},
  year={2021}
}
```

### Limitation Concerning Underlying Contract Rights
> [!IMPORTANT]
> The underlying commercial contracts in CUAD originate from public **U.S. Securities and Exchange Commission (SEC) EDGAR** filings. The Atticus Project and this project do not hold underlying copyright to the commercial contracts themselves. Annotations are provided under CC BY 4.0 for research and compliance auditing purposes.

---

## 3. Directory Structure

```text
Autonomous-Compliance-and-Document-Risk-Auditor/
├── .gitignore                      # Excludes raw data, virtualenvs, cache, and reports
├── README.md                       # Complete documentation and execution guide
├── configs/
│   └── cuad_categories.yaml        # Configuration for the 15 selected contractual attributes
├── data/
│   ├── raw/
│   │   └── cuad/                   # Stored dataset files (git-ignored)
│   │       ├── CUAD_v1.zip         # Official archive
│   │       ├── CUAD_v1.json        # SQuAD 2.0 format annotations
│   │       ├── master_clauses.csv  # Tabular clause matrix
│   │       └── manifest.json       # Provenance, SHA-256 checksums, and license
│   └── processed/
│       └── splits.json             # Deterministic document-level train/val/test splits
├── reports/
│   └── evaluation/
│       └── cuad/                   # Machine-readable JSON and CSV evaluation reports
├── src/
│   ├── data/
│   │   ├── downloader.py           # Automated verified dataset download with SHA-256
│   │   ├── loader.py               # Robust CUAD parser (SQuAD JSON & CSV), inference isolation
│   │   ├── schema.py               # Pydantic data models (Contracts, Annotations, Spans, Manifest)
│   │   └── splits.py               # Deterministic document-level train/val/test partitioner
│   ├── extraction/
│   │   ├── interface.py            # BaseExtractor abstract base class
│   │   ├── baseline.py             # Rule-based / heuristic baseline extractor
│   │   ├── llm.py                  # Configurable LLM extractor (reads env vars, rejects unconfigured)
│   │   ├── normalizer.py           # Date, party, boolean, and token normalizers
│   │   └── schema.py               # ExtractionResult & ExtractionStatus Pydantic models
│   └── evaluation/
│       ├── metrics.py              # Precision, Recall, F1, Exact Match, Jaccard IoU
│       └── evaluator.py            # End-to-end evaluation runner and report generator
├── scripts/
│   ├── download_cuad.py            # CLI script to download and verify dataset
│   └── evaluate_extraction.py      # CLI script to run extraction benchmarks
└── tests/
    ├── conftest.py                 # Isolated test fixtures (mini SQuAD JSON & CSV)
    ├── test_categories.py          # Category YAML configuration and alias resolution tests
    ├── test_downloader.py          # Download integrity and checksum tests
    ├── test_evaluator.py           # End-to-end evaluation runner tests
    ├── test_extraction_schema.py   # Pydantic schema validation & NOT_FOUND handling tests
    ├── test_loader.py              # Parsing, missing annotations, multi-spans, and leakage tests
    ├── test_metrics.py             # Precision, recall, F1, exact match, and Jaccard tests
    ├── test_normalizer.py          # Text, date, party, and boolean normalizer tests
    └── test_splits.py              # Document-level splitting and seed reproducibility tests
```

---

## 4. The 15 Selected Contractual Attributes

| # | Category Name | Output Type | Classification | Meaning & Legal Purpose |
|---|---|---|---|---|
| 1 | **Parties** | `list[str]` | Value Extraction | Legal names of all signatory entities/individuals entering into the contract. |
| 2 | **Agreement Date** | `date` | Value Extraction | The formal date when the contract was executed or entered into. |
| 3 | **Effective Date** | `date` | Value Extraction | Date on which contractual rights and obligations take legal effect. |
| 4 | **Expiration Date** | `date_or_duration` | Value Extraction | Date or duration when the initial term expires (or stated as Perpetual). |
| 5 | **Renewal Term** | `duration` | Value Extraction | Duration of subsequent terms after the initial term expires. |
| 6 | **Notice Period to Terminate Renewal** | `duration` | Value Extraction | Required advance notice window to prevent automatic renewal. |
| 7 | **Governing Law** | `string` | Value Extraction | Jurisdiction (State, Province, Country) governing interpretation and dispute resolution. |
| 8 | **Termination for Convenience** | `boolean` | Presence / Absence | Unilateral right of a party to terminate without cause upon notice. |
| 9 | **Cap on Liability** | `presence_or_value` | Presence / Value | Monetary ceiling or formula limiting damages in breach of contract claims. |
| 10 | **Insurance** | `boolean` | Presence / Absence | Affirmative obligation to maintain specific commercial insurance coverages. |
| 11 | **Audit Rights** | `boolean` | Presence / Absence | Right to inspect and audit counterparty books, records, facilities, or logs. |
| 12 | **Uncapped Liability** | `boolean` | Presence / Absence | Express carve-outs (e.g. IP infringement, gross negligence) where liability is unlimited. |
| 13 | **Liquidated Damages** | `presence_or_value` | Presence / Value | Pre-agreed damages or termination fees payable upon specified breaches or early termination. |
| 14 | **Anti-Assignment** | `boolean` | Presence / Absence | Requirement of prior consent/notice to assign or transfer the contract to a third party. |
| 15 | **License Grant** | `boolean` | Presence / Absence | Express grant of rights/license to intellectual property, software, or technology. |

---

## 5. Setup & Installation

### Windows (PowerShell)
```powershell
# 1. Clone repository
git clone https://github.com/kesshar/Autonomous-Compliance-and-Document-Risk-Auditor.git
cd Autonomous-Compliance-and-Document-Risk-Auditor

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install core dependencies
pip install pydantic pyyaml pytest requests pandas numpy python-dateutil scikit-learn
```

### Linux / WSL Ubuntu
```bash
# 1. Clone repository
git clone https://github.com/kesshar/Autonomous-Compliance-and-Document-Risk-Auditor.git
cd Autonomous-Compliance-and-Document-Risk-Auditor

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install core dependencies
pip install pydantic pyyaml pytest requests pandas numpy python-dateutil scikit-learn
```

---

## 6. How to Download the Dataset

The dataset download mechanism streams the official archive from Zenodo (or GitHub/HuggingFace), validates SHA-256 checksums, unpacks to `data/raw/cuad/`, and records a signed `manifest.json`.

```powershell
# Windows PowerShell
python scripts/download_cuad.py --source zenodo

# Force re-download if existing files need refreshing
python scripts/download_cuad.py --source zenodo --force
```

```bash
# Linux / WSL Ubuntu
python3 scripts/download_cuad.py --source zenodo
```

---

## 7. How to Execute the Loader in Python

The loader provides an inference view that **strictly prevents gold annotation leakage**:

```python
from src.data.loader import CUADLoader
from src.data.splits import create_document_splits, filter_documents_by_split

# 1. Initialize loader with the 15 selected categories
loader = CUADLoader(config_path="configs/cuad_categories.yaml")

# 2. Parse SQuAD-style JSON annotations
documents = loader.parse_squad_json("data/raw/cuad/CUAD_v1.json")
print(f"Loaded {len(documents)} contracts.")

# 3. Partition documents into deterministic splits (seed=42)
manifest = create_document_splits(list(documents.keys()))
test_docs = filter_documents_by_split(documents, manifest, "test")
print(f"Test split contains {len(test_docs)} contracts.")

# 4. Safe inference view (ZERO gold annotations exposed to model)
sample_doc = next(iter(test_docs.values()))
unlabeled_context = sample_doc.get_unlabeled_context()
# unlabeled_context contains: id, title, text, length_chars (NO answers or spans!)
```

---

## 8. How to Run the Evaluation Script

### Running with the Offline Deterministic Baseline
```powershell
# Evaluate across the test split contracts
python scripts/evaluate_extraction.py --extractor baseline --split test

# Evaluate on a quick sample of 10 contracts
python scripts/evaluate_extraction.py --extractor baseline --max-docs 10
```

### Running with a Configured LLM Provider
```powershell
# Set environment variables (never hardcoded in source code!)
$env:LLM_PROVIDER="openai"
$env:OPENAI_API_KEY="sk-..."
$env:LLM_MODEL="gpt-4o-mini"

# Execute evaluation
python scripts/evaluate_extraction.py --extractor llm --split test
```
*Note: If `LLM_PROVIDER` or required API keys are not configured in your environment, the evaluation script safely halts and informs you that model evaluation cannot run, rather than fabricating placeholder accuracy numbers.*

---

## 9. Understanding Evaluation Metrics

The evaluation outputs three complementary metric views:
1. **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$ — Measures extraction purity: when the system claims a clause exists or extracts a value, how often is it correct?
2. **Recall**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$ — Measures completeness: out of all contractual clauses that actually exist in the contract, how many were discovered?
3. **F1-Score**: $\frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ — Harmonic mean balancing false positives and false negatives.
4. **Exact Match (EM)**: Proportion of extracted answers that match gold annotations exactly after domain normalization (date standardization, legal corporate form trimming, whitespace collapse).
5. **Jaccard IoU**: Token-level Intersection-Over-Union between gold evidence text and extracted text (standard threshold = 0.50).
6. **Macro vs. Micro Averages**:
   - **Macro Average**: Computes metrics for each category independently and takes the unweighted arithmetic mean, ensuring rare clauses (e.g., Liquidated Damages) are weighted equally with frequent clauses.
   - **Micro Average**: Aggregates all global true positives, false positives, and false negatives across all categories combined.

### Evaluation Reports
Results are automatically archived in `reports/evaluation/cuad/`:
- `cuad_baseline_report_<timestamp>.json`: Comprehensive machine-readable metrics and metadata.
- `cuad_baseline_summary_<timestamp>.csv`: Summary table with category-by-category rows, macro, and micro averages.
- `cuad_baseline_errors_<timestamp>.json`: Categorized false-positive, false-negative, and misaligned extraction examples for diagnostic debugging.

---

## 10. Running the Test Suite

```powershell
# Windows PowerShell
python -m pytest tests/ -v
```

```bash
# Linux / WSL Ubuntu
pytest tests/ -v
```

All 31 test cases cover:
- Dataset loading and schema validation
- Missing annotations & multi-span handling
- Category mapping and alias resolution
- Normalization (dates, booleans, parties, text)
- Deterministic document splitting (zero leakage)
- Precision, recall, and F1 calculations
- Malformed record resilience
- Pydantic extraction output validation

