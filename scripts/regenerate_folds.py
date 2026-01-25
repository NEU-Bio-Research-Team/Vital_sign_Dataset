#!/usr/bin/env python3
"""Regenerate folds from existing preprocessed data."""

import argparse
import sys
from pathlib import Path

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from vitaldb_aki.config import load_config
from vitaldb_aki.data.preprocessing import build_aki_labels, build_manifest, create_folds
from vitaldb_aki.utils.paths import get_paths


def main():
    """Regenerate folds from existing data."""
    parser = argparse.ArgumentParser(description="Regenerate CV folds from existing data")
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
    args = parser.parse_args()

    # Load config
    config = load_config(args.config, args.experiment_name)
    print(f"Using experiment: {args.experiment_name}")
    print(f"Artifacts directory: {config.artifacts_dir}")

    paths = get_paths(config)

    # Load existing data
    print("\n=== Loading existing data ===")
    df_labels = build_aki_labels(config, force=False)
    df_manifest = build_manifest(config, df_labels, force=False)
    
    # Load df_usable
    usable_path = paths["artifacts_dir"] / "df_usable.csv"
    if not usable_path.exists():
        raise FileNotFoundError(
            f"df_usable.csv not found at {usable_path}. "
            f"Please run preprocessing first: python scripts/preprocess.py --experiment-name {args.experiment_name}"
        )
    df_usable = pd.read_csv(usable_path)
    print(f"Loaded {len(df_usable)} usable cases")
    
    # Filter to only cases that have NPZ files
    cache_dir = paths["cache_dir"]
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"Cache directory not found at {cache_dir}. "
            f"Please run preprocessing first: python scripts/preprocess.py --experiment-name {args.experiment_name}"
        )
    
    # Get all existing NPZ files
    existing_npz = set()
    for npz_file in cache_dir.glob("case_*.npz"):
        try:
            caseid = int(npz_file.stem.split("_")[1])
            existing_npz.add(caseid)
        except (ValueError, IndexError):
            continue
    
    print(f"Found {len(existing_npz)} NPZ files in cache")
    
    if len(existing_npz) == 0:
        raise FileNotFoundError(
            f"No NPZ files found in {cache_dir}. "
            f"Please run preprocessing to generate NPZ files: "
            f"python scripts/preprocess.py --experiment-name {args.experiment_name}"
        )
    
    # Filter df_usable to only cases with NPZ files
    df_usable = df_usable[df_usable["caseid"].isin(existing_npz)].copy()
    print(f"Filtered to {len(df_usable)} cases with NPZ files")

    # Regenerate folds
    print("\n=== Regenerating folds ===")
    folds, df_master = create_folds(config, df_labels, df_manifest, df_usable, force=True)

    print("\n=== Folds regenerated ===")
    print(f"Total cases: {len(df_master)}")
    print(f"Folds created: {len(folds)}")
    for fold in folds:
        print(
            f"  Fold {fold['fold']}: train={fold['n_train']}, val={fold['n_val']}, "
            f"train_pos={fold['train_pos']}, val_pos={fold['val_pos']}"
        )


if __name__ == "__main__":
    main()

