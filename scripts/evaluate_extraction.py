#!/usr/bin/env python3
"""CLI script to evaluate contract extraction pipelines against CUAD ground truth annotations."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import CUADLoader
from src.data.splits import load_splits, filter_documents_by_split
from src.evaluation.evaluator import ExtractionEvaluator
from src.extraction.baseline import RuleBasedBaselineExtractor
from src.extraction.llm import LLMExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate contract clause extraction against official CUAD annotations."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/cuad/CUAD_v1.json"),
        help="Path to CUAD dataset (SQuAD JSON or master_clauses.csv). Default: data/raw/cuad/CUAD_v1.json",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/cuad_categories.yaml"),
        help="Path to categories YAML config. Default: configs/cuad_categories.yaml",
    )
    parser.add_argument(
        "--extractor",
        choices=["baseline", "llm"],
        default="baseline",
        help="Extractor implementation to benchmark ('baseline' or 'llm'). Default: baseline",
    )
    parser.add_argument(
        "--split",
        choices=["test", "val", "train", "all"],
        default="test",
        help="Document split to evaluate on. Default: test",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/processed/splits.json"),
        help="Path to split manifest JSON file. Default: data/processed/splits.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/evaluation/cuad"),
        help="Directory to save evaluation JSON and CSV reports. Default: reports/evaluation/cuad",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit the number of evaluated documents (useful for quick diagnostics).",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.5,
        help="Jaccard IoU threshold for evidence span matching (default: 0.5).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed logging output.",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # 1. Initialize Extractor
    if args.extractor == "llm":
        extractor = LLMExtractor()
        is_ok, msg = extractor.check_configuration()
        if not is_ok:
            print("\n[ERROR] Model evaluation cannot run because LLM provider is not configured.", file=sys.stderr)
            print(f"Details: {msg}", file=sys.stderr)
            print("Please configure environment variables (e.g., LLM_PROVIDER, OPENAI_API_KEY) or run with '--extractor baseline'.", file=sys.stderr)
            sys.exit(1)
    else:
        extractor = RuleBasedBaselineExtractor()

    # 2. Check Data Path
    data_path = args.data_path
    if not data_path.exists():
        # Check alternative common locations
        alt_paths = [
            Path("data/raw/cuad/CUAD_v1/CUAD_v1.json"),
            Path("data/raw/cuad/test.json"),
            Path("data/raw/cuad/master_clauses.csv"),
        ]
        for ap in alt_paths:
            if ap.exists():
                data_path = ap
                print(f"[INFO] Using detected dataset path: {data_path}")
                break
        else:
            print(f"\n[ERROR] CUAD dataset file not found at: {data_path}", file=sys.stderr)
            print("Run 'python scripts/download_cuad.py' first to acquire the dataset.", file=sys.stderr)
            sys.exit(1)

    # 3. Load Target Categories
    import yaml
    with open(args.config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    target_categories: List[str] = [c["name"] for c in config_data.get("categories", [])]

    # 4. Load Dataset Documents
    loader = CUADLoader(target_categories=target_categories, config_path=args.config_path)
    if data_path.suffix.lower() == ".json":
        documents = loader.parse_squad_json(data_path, max_documents=args.max_docs)
    elif data_path.suffix.lower() == ".csv":
        documents = loader.parse_master_clauses_csv(data_path, max_documents=args.max_docs)
    else:
        print(f"[ERROR] Unsupported file extension for dataset: {data_path.suffix}", file=sys.stderr)
        sys.exit(1)

    if not documents:
        print(f"[ERROR] No documents loaded from {data_path}.", file=sys.stderr)
        sys.exit(1)

    # 5. Filter by Split if requested
    if args.split != "all" and args.split_manifest.exists():
        print(f"[INFO] Filtering documents using split '{args.split}' from {args.split_manifest}...")
        manifest = load_splits(args.split_manifest)
        documents = filter_documents_by_split(documents, manifest, args.split)
        print(f"[INFO] {len(documents)} documents retained in '{args.split}' split.")
    elif args.split != "all" and not args.split_manifest.exists():
        print(f"[WARN] Split manifest {args.split_manifest} not found. Evaluating on all {len(documents)} loaded documents.")

    # 6. Run Evaluation
    evaluator = ExtractionEvaluator(
        extractor=extractor,
        category_config_path=args.config_path,
        iou_threshold=args.iou_thresh,
    )

    print(f"\nRunning extraction evaluation on {len(documents)} contracts across {len(target_categories)} categories...")
    report = evaluator.evaluate_documents(documents, target_categories)

    # 7. Save Reports
    paths = evaluator.save_reports(report, output_dir=args.output_dir, prefix=f"cuad_{args.extractor}")

    # 8. Print Console Summary
    print("\n" + "=" * 92)
    print("                      CUAD EXTRACTION BENCHMARK RESULTS")
    print("=" * 92)
    header = f"{'Category':<35} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'ExactM':<7} | {'IoU':<7} | {'Gold':<5} | {'Pred':<5}"
    print(header)
    print("-" * 92)

    for cat_name, cm in report.categories.items():
        row = (
            f"{cat_name:<35} | "
            f"{cm.precision:<7.4f} | "
            f"{cm.recall:<7.4f} | "
            f"{cm.f1:<7.4f} | "
            f"{cm.exact_match_ratio:<7.4f} | "
            f"{cm.avg_jaccard_iou:<7.4f} | "
            f"{cm.total_gold_instances:<5} | "
            f"{cm.total_predictions:<5}"
        )
        print(row)

    print("-" * 92)
    agg = report.aggregate_metrics
    print(
        f"{'--- MACRO AVERAGE ---':<35} | "
        f"{agg.macro_precision:<7.4f} | "
        f"{agg.macro_recall:<7.4f} | "
        f"{agg.macro_f1:<7.4f} | "
        f"{'-':<7} | {'-':<7} | {'-':<5} | {'-':<5}"
    )
    print(
        f"{'--- MICRO AVERAGE ---':<35} | "
        f"{agg.micro_precision:<7.4f} | "
        f"{agg.micro_recall:<7.4f} | "
        f"{agg.micro_f1:<7.4f} | "
        f"{'-':<7} | {'-':<7} | {'-':<5} | {'-':<5}"
    )
    print("=" * 92)
    print(f"\nEvaluation completed successfully. Reports saved to:")
    print(f"  JSON Report: {paths['json_report']}")
    print(f"  CSV Summary: {paths['csv_summary']}")
    print(f"  Error Log:   {paths['error_examples']}\n")


if __name__ == "__main__":
    main()

