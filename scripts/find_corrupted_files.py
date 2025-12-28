#!/usr/bin/env python3
"""Find and handle corrupted NPZ files."""

import json
import sys
from pathlib import Path

import numpy as np


def find_corrupted_files(cache_dir: Path) -> list:
    """Find all corrupted NPZ files."""
    corrupted = []
    
    for npz_file in sorted(cache_dir.glob("case_*.npz")):
        try:
            # Check file size first
            if npz_file.stat().st_size < 100:
                corrupted.append(npz_file)
                continue
            
            # Try to load
            data = np.load(npz_file, allow_pickle=True)
            if "x" not in data or "valid_len" not in data:
                corrupted.append(npz_file)
        except Exception as e:
            corrupted.append(npz_file)
    
    return corrupted


def remove_from_folds(folds_path: Path, caseids: list):
    """Remove case IDs from folds."""
    with open(folds_path, "r") as f:
        folds = json.load(f)
    
    print(f"\nRemoving {len(caseids)} cases from folds...")
    for fold in folds:
        updated = False
        for caseid in caseids:
            if caseid in fold.get("train_caseids", []):
                fold["train_caseids"] = [c for c in fold["train_caseids"] if c != caseid]
                updated = True
            if caseid in fold.get("val_caseids", []):
                fold["val_caseids"] = [c for c in fold["val_caseids"] if c != caseid]
                updated = True
        
        if updated:
            fold["n_train"] = len(fold["train_caseids"])
            fold["n_val"] = len(fold["val_caseids"])
            print(f"  Updated fold {fold['fold']}: train={fold['n_train']}, val={fold['n_val']}")
    
    with open(folds_path, "w") as f:
        json.dump(folds, f, indent=2)
    print("✓ Updated folds.json")


def main():
    """Main function."""
    artifacts_dir = Path("artifacts/demo_5signals")
    cache_dir = artifacts_dir / "cache_npz"
    folds_path = artifacts_dir / "folds.json"
    
    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        sys.exit(1)
    
    # Find corrupted files
    print("Scanning for corrupted NPZ files...")
    corrupted = find_corrupted_files(cache_dir)
    
    if not corrupted:
        print("✓ No corrupted files found")
        return
    
    print(f"\nFound {len(corrupted)} corrupted files:")
    caseids = []
    for f in corrupted:
        caseid = int(f.stem.split("_")[1])
        caseids.append(caseid)
        print(f"  case_{caseid}.npz ({f.stat().st_size} bytes)")
    
    # Ask what to do
    print(f"\nOptions:")
    print("  1. Remove corrupted files and remove from folds")
    print("  2. Just remove from folds (keep files)")
    print("  3. Exit without changes")
    
    choice = input("\nEnter choice (1-3) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        # Remove files
        for f in corrupted:
            f.unlink()
        print(f"\n✓ Removed {len(corrupted)} corrupted files")
        
        # Remove from folds
        if folds_path.exists():
            remove_from_folds(folds_path, caseids)
        else:
            print("Warning: folds.json not found, cannot update folds")
    
    elif choice == "2":
        # Just remove from folds
        if folds_path.exists():
            remove_from_folds(folds_path, caseids)
        else:
            print("Error: folds.json not found")
            sys.exit(1)
    
    elif choice == "3":
        print("Exiting without changes")
        return
    
    else:
        print(f"Invalid choice: {choice}")
        sys.exit(1)
    
    print(f"\n✓ Done! You can now:")
    print(f"  - Run training: python scripts/train.py --experiment-name demo_5signals")
    print(f"  - Regenerate files: python scripts/preprocess.py --experiment-name demo_5signals --force")


if __name__ == "__main__":
    main()

