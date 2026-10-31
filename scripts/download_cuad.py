#!/usr/bin/env python3
"""CLI script to acquire, verify, and catalog the CUAD v1 dataset."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader import acquire_cuad_dataset, OFFICIAL_ZENODO_URL, OFFICIAL_GITHUB_DATA_URL


def main():
    parser = argparse.ArgumentParser(description="Download and verify the official CUAD v1 dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/cuad"),
        help="Directory to store dataset files (default: data/raw/cuad)",
    )
    parser.add_argument(
        "--source",
        choices=["zenodo", "github"],
        default="zenodo",
        help="Source provider for dataset archive (default: zenodo)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    url = OFFICIAL_ZENODO_URL if args.source == "zenodo" else OFFICIAL_GITHUB_DATA_URL

    print(f"Starting acquisition of CUAD dataset from {args.source}...")
    print(f"Destination: {args.output_dir.resolve()}")
    try:
        manifest = acquire_cuad_dataset(output_dir=args.output_dir, source_url=url, force=args.force)
        print("\n=== CUAD Dataset Acquisition Successful ===")
        print(f"Version:      {manifest.dataset_version}")
        print(f"Source:       {manifest.source_url}")
        print(f"Timestamp:    {manifest.download_timestamp}")
        print(f"Archive SHA:  {manifest.sha256_checksum}")
        print(f"Files Tracked:{len(manifest.files)}")
        print("\nLicense & Legal Notice:")
        print(f"{manifest.license}")
        print(f"\nAttribution:\n{manifest.attribution}")
        print(f"\nRights Limitation:\n{manifest.rights_limitation}")
    except Exception as e:
        print(f"\nError acquiring dataset: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

