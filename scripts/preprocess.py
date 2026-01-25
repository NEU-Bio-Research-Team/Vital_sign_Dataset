#!/usr/bin/env python3
"""Preprocessing script for VitalDB AKI prediction."""

import argparse
import sys
from pathlib import Path

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vitaldb_aki.config import load_config
from vitaldb_aki.data.preprocessing import (
    build_aki_labels,
    build_manifest,
    ingest_tracks,
    create_folds,
    fit_scalers,
)
from vitaldb_aki.utils.paths import setup_directories


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Preprocess VitalDB data for AKI prediction")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file (default: use default config)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="new_optional_exp",
        help="Experiment name (default: new_optional_exp)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if artifacts exist",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(
        args.config,
        args.experiment_name,
        prefer_saved_config=not args.force,
    )
    print(f"Using experiment: {args.experiment_name}")
    print(f"Artifacts directory: {config.artifacts_dir}")
    print(f"Signals: {list(config.signals)}")
    print(f"Required signals: {list(config.required_signals)}")
    print(f"include_optional_signals: {config.include_optional_signals}")

    # Setup directories
    paths = setup_directories(config)
    config.save(paths["artifacts_dir"] / "config.json")

    # Step 1: Build AKI labels
    print("\n=== Step 1: Building AKI labels ===")
    df_labels = build_aki_labels(config, force=args.force)

    # Step 2: Build manifest
    print("\n=== Step 2: Building manifest ===")
    df_manifest = build_manifest(config, df_labels, force=args.force)

    # Step 3: Ingest tracks
    print("\n=== Step 3: Ingesting tracks ===")
    df_usable, df_failed = ingest_tracks(config, df_manifest, force=args.force)

    # Step 4: Create folds
    print("\n=== Step 4: Creating folds ===")
    folds, df_master = create_folds(config, df_labels, df_manifest, df_usable, force=args.force)

    # Step 5: Fit scalers
    print("\n=== Step 5: Fitting scalers ===")
    fold_scalers = fit_scalers(config, folds, force=args.force)

    print("\n=== Preprocessing complete ===")
    print(f"Usable cases: {len(df_usable)}")
    print(f"Failed cases: {len(df_failed)}")
    print(f"Folds created: {len(folds)}")


if __name__ == "__main__":
    main()

