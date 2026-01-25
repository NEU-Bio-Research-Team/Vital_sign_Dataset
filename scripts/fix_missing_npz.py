#!/usr/bin/env python3
"""Remove cases from folds that don't have NPZ files."""

import json
from pathlib import Path


def main():
    """Remove cases without NPZ files from folds."""
    artifacts_dir = Path("artifacts/new_optional_exp")
    cache_dir = artifacts_dir / "cache_npz"
    folds_path = artifacts_dir / "folds.json"
    
    if not folds_path.exists():
        print(f"folds.json not found at {folds_path}")
        return
    
    if not cache_dir.exists():
        print(f"Cache directory not found at {cache_dir}")
        return
    
    # Get all existing NPZ files
    existing_npz = set()
    for npz_file in cache_dir.glob("case_*.npz"):
        try:
            caseid = int(npz_file.stem.split("_")[1])
            existing_npz.add(caseid)
        except (ValueError, IndexError):
            continue
    
    print(f"Found {len(existing_npz)} NPZ files in cache")
    
    # Load folds
    with open(folds_path, "r") as f:
        folds = json.load(f)
    
    # Check which cases are missing
    all_caseids = set()
    for fold in folds:
        all_caseids.update(fold.get("train_caseids", []))
        all_caseids.update(fold.get("val_caseids", []))
    
    missing = sorted(all_caseids - existing_npz)
    
    if not missing:
        print("✓ All cases in folds have NPZ files")
        return
    
    print(f"\nFound {len(missing)} cases in folds without NPZ files:")
    for caseid in missing[:20]:  # Show first 20
        print(f"  case_{caseid}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")
    
    # Remove missing cases from folds
    print(f"\nRemoving {len(missing)} cases from folds...")
    for fold in folds:
        updated = False
        train_before = len(fold.get("train_caseids", []))
        val_before = len(fold.get("val_caseids", []))
        
        fold["train_caseids"] = [c for c in fold.get("train_caseids", []) if c in existing_npz]
        fold["val_caseids"] = [c for c in fold.get("val_caseids", []) if c in existing_npz]
        
        fold["n_train"] = len(fold["train_caseids"])
        fold["n_val"] = len(fold["val_caseids"])
        
        train_after = fold["n_train"]
        val_after = fold["n_val"]
        
        if train_before != train_after or val_before != val_after:
            updated = True
            print(
                f"  Fold {fold['fold']}: "
                f"train {train_before}→{train_after}, "
                f"val {val_before}→{val_after}"
            )
    
    # Save updated folds
    with open(folds_path, "w") as f:
        json.dump(folds, f, indent=2)
    
    print(f"\n✓ Updated folds.json")
    print(f"  Removed {len(missing)} cases without NPZ files")
    print(f"  Remaining cases: {len(existing_npz)}")
    
    # Verify no missing cases remain
    all_caseids_after = set()
    for fold in folds:
        all_caseids_after.update(fold.get("train_caseids", []))
        all_caseids_after.update(fold.get("val_caseids", []))
    
    still_missing = sorted(all_caseids_after - existing_npz)
    if still_missing:
        print(f"\n⚠ Warning: {len(still_missing)} cases still missing NPZ files!")
    else:
        print("\n✓ All cases in folds now have NPZ files")


if __name__ == "__main__":
    main()

