#!/usr/bin/env python3
"""
================================================================================
Creates the official final submission zip packages:
1. Brad_Pitt_submission.zip (Official Registered Team Name on Unstop)
2. PrecisionResolvers_submission.zip (Alternative submission archive)
================================================================================
Contains:
1. output/matching_results.tsv (V19 finetuned LightGBM results)
2. output/candidate_pairs.tsv (100% consistent with matching_results.tsv)
3. code/business_entity_resolution/ (runnable pipeline)
4. Documentation_template.md & documentation.md (comprehensive methodology report)
================================================================================
"""

import sys
import os
import shutil
import zipfile
import time
from pathlib import Path

BASE_DIR = Path(r".")
DESKTOP_DIR = Path(r"~/Desktop")

FILES_TO_ADD = [
    (BASE_DIR / "output" / "matching_results.tsv", "output/matching_results.tsv"),
    (BASE_DIR / "output" / "candidate_pairs.tsv", "output/candidate_pairs.tsv"),
    (BASE_DIR / "Documentation_template.md", "Documentation_template.md"),
    (BASE_DIR / "documentation.md", "documentation.md"),
]

CODE_DIR = BASE_DIR / "code" / "business_entity_resolution"

def build_zip(zip_path):
    t0 = time.time()
    print(f"\nBuilding {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src, arc in FILES_TO_ADD:
            print(f"  Adding {arc} ({src.stat().st_size / 1024 / 1024:.1f} MB)...")
            zf.write(src, arc)
            
        print("  Adding code/business_entity_resolution/...")
        for p in CODE_DIR.rglob("*"):
            if p.is_file() and "__pycache__" not in str(p) and not p.name.endswith(".pyc"):
                rel_path = p.relative_to(BASE_DIR)
                zf.write(p, str(rel_path).replace("\\", "/"))
                
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"  Successfully created {zip_path.name} ({size_mb:.1f} MB) in {time.time()-t0:.1f}s.")

def main():
    print("=" * 70)
    print("PACKAGING FINAL AMAZON ML CHALLENGE SUBMISSION")
    print("=" * 70)
    
    # 1. Verify required files exist
    for src, arc in FILES_TO_ADD:
        if not src.exists():
            print(f"ERROR: Missing required file {src}")
            sys.exit(1)
            
    if not CODE_DIR.exists():
        print(f"ERROR: Missing code directory {CODE_DIR}")
        sys.exit(1)
        
    # 2. Build Brad_Pitt_submission.zip (in project folder and on Desktop)
    bp_proj = BASE_DIR / "Brad_Pitt_submission.zip"
    build_zip(bp_proj)
    shutil.copyfile(bp_proj, DESKTOP_DIR / "Brad_Pitt_submission.zip")
    print(f"Copied to: {DESKTOP_DIR / 'Brad_Pitt_submission.zip'}")
    
    # 3. Build PrecisionResolvers_submission.zip
    pr_proj = BASE_DIR / "PrecisionResolvers_submission.zip"
    build_zip(pr_proj)
    shutil.copyfile(pr_proj, DESKTOP_DIR / "PrecisionResolvers_submission.zip")
    print(f"Copied to: {DESKTOP_DIR / 'PrecisionResolvers_submission.zip'}")
    
    print("\n" + "=" * 70)
    print("All submission packages successfully updated and verified!")
    print("=" * 70)

if __name__ == "__main__":
    main()
