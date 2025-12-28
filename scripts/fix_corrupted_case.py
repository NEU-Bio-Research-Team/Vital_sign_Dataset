#!/usr/bin/env python3
"""Remove corrupted case from dataset."""

import json
import sys
from pathlib import Path

import pandas as pd

CORRUPTED_CASEID = 16


def main():
    """Remove corrupted case from all relevant files."""
    artifacts_dir = Path("artifacts/demo_5signals")
    
    # Remove from df_usable
    usable_path = artifacts_dir / "df_usable.csv"
    if usable_path.exists():
        df_usable = pd.read_csv(usable_path)
        print(f"Before: {len(df_usable)} usable cases")
        df_usable = df_usable[df_usable["caseid"] != CORRUPTED_CASEID]
        df_usable.to_csv(usable_path, index=False)
        print(f"After: {len(df_usable)} usable cases (removed case_{CORRUPTED_CASEID})")
    
    # Remove from cohort_master
    master_path = artifacts_dir / "cohort_master.csv"
    if master_path.exists():
        df_master = pd.read_csv(master_path)
        print(f"Before: {len(df_master)} cases in master")
        df_master = df_master[df_master["caseid"] != CORRUPTED_CASEID]
        df_master.to_csv(master_path, index=False)
        print(f"After: {len(df_master)} cases in master (removed case_{CORRUPTED_CASEID})")
    
    # Update folds
    folds_path = artifacts_dir / "folds.json"
    if folds_path.exists():
        with open(folds_path, "r") as f:
            folds = json.load(f)
        
        for fold in folds:
            updated = False
            if CORRUPTED_CASEID in fold.get("train_caseids", []):
                fold["train_caseids"] = [c for c in fold["train_caseids"] if c != CORRUPTED_CASEID]
                fold["n_train"] = len(fold["train_caseids"])
                updated = True
            if CORRUPTED_CASEID in fold.get("val_caseids", []):
                fold["val_caseids"] = [c for c in fold["val_caseids"] if c != CORRUPTED_CASEID]
                fold["n_val"] = len(fold["val_caseids"])
                updated = True
            if updated:
                print(f"Updated fold {fold['fold']} to remove case_{CORRUPTED_CASEID}")
        
        with open(folds_path, "w") as f:
            json.dump(folds, f, indent=2)
        print("Updated folds.json")
    
    print(f"\n✓ Successfully removed case_{CORRUPTED_CASEID} from dataset")


if __name__ == "__main__":
    main()

