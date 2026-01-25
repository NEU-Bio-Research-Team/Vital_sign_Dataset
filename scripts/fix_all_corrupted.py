#!/usr/bin/env python3
"""Automatically find and fix all corrupted NPZ files."""

import json
from pathlib import Path

import numpy as np


def main():
    """Find and fix all corrupted files automatically."""
    artifacts_dir = Path("artifacts/new_optional_exp")
    cache_dir = artifacts_dir / "cache_npz"
    folds_path = artifacts_dir / "folds.json"
    
    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        return
    
    # Find corrupted files
    print("Scanning for corrupted NPZ files...")
    corrupted = []
    caseids = []
    
    for npz_file in sorted(cache_dir.glob("case_*.npz")):
        try:
            # Check file size first
            if npz_file.stat().st_size < 100:
                corrupted.append(npz_file)
                caseid = int(npz_file.stem.split("_")[1])
                caseids.append(caseid)
                continue
            
            # Try to load
            data = np.load(npz_file, allow_pickle=True)
            if "x" not in data or "valid_len" not in data:
                corrupted.append(npz_file)
                caseid = int(npz_file.stem.split("_")[1])
                caseids.append(caseid)
        except Exception as e:
            corrupted.append(npz_file)
            caseid = int(npz_file.stem.split("_")[1])
            caseids.append(caseid)
    
    if not corrupted:
        print("✓ No corrupted files found")
        return
    
    print(f"\nFound {len(corrupted)} corrupted files:")
    for f, caseid in zip(corrupted, caseids):
        print(f"  case_{caseid}.npz ({f.stat().st_size} bytes)")
    
    # Remove corrupted files
    for f in corrupted:
        f.unlink()
    print(f"\n✓ Removed {len(corrupted)} corrupted files")
    
    # Remove from folds
    if folds_path.exists():
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
    else:
        print("Warning: folds.json not found, cannot update folds")
    
    print(f"\n✓ Done! You can now run training:")
    print(f"  python scripts/train.py --experiment-name new_optional_exp")


if __name__ == "__main__":
    main()

