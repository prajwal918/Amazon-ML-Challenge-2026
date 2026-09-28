#!/usr/bin/env python3
"""
Kaggle Cloud Queue Manager & Auto-Downloader
Monitors the 5 active Kaggle kernels.
When one finishes:
  1. Downloads the output results to local project & Desktop
  2. Packages the submission ZIP
  3. Automatically pushes the next queued model (V30 -> V40)
Zero CPU usage (sleeps 60s between API status calls).
"""

import os, sys, time, shutil, subprocess, zipfile
from pathlib import Path

BASE_DIR = Path(r".")
DESKTOP = Path(r"~/Desktop")

ACTIVE_KERNELS = [
    "node_alpha/amazon-ml-zero-dep-grandmaster",
    "node_alpha/amazon-ml-v26-ultra-precision",
    "node_alpha/amazon-ml-v27-gpu-accelerated",
    "node_alpha/amazon-ml-v28-tri-branch-ranker",
    "node_alpha/amazon-ml-v29-deep-tree"
]

PENDING_QUEUE = [
    f"v{i}" for i in range(30, 41)
]

def check_status(kernel_slug):
    cmd = ["python", "-m", "kaggle", "kernels", "status", kernel_slug]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    if "KernelWorkerStatus.COMPLETE" in out: return "COMPLETE"
    if "KernelWorkerStatus.RUNNING" in out: return "RUNNING"
    if "KernelWorkerStatus.ERROR" in out: return "ERROR"
    if "KernelWorkerStatus.QUEUED" in out: return "QUEUED"
    return "UNKNOWN"

def download_and_package(kernel_slug):
    slug_name = kernel_slug.split("/")[-1]
    out_dir = BASE_DIR / f"kaggle_output_{slug_name}"
    out_dir.mkdir(exist_ok=True)
    print(f"\nDownloading output for {slug_name}...")
    cmd = ["python", "-m", "kaggle", "kernels", "output", kernel_slug, "-p", str(out_dir)]
    subprocess.run(cmd, check=False)
    
    # Check for matching_results.tsv
    matching_tsv = None
    for p in out_dir.rglob("matching_results*.tsv"):
        matching_tsv = p; break
        
    if matching_tsv and matching_tsv.exists():
        print(f"  Found prediction file: {matching_tsv.name} ({matching_tsv.stat().st_size / 1024 / 1024:.1f} MB)")
        # Copy to output/
        dest_tsv = BASE_DIR / "output" / "matching_results.tsv"
        shutil.copyfile(matching_tsv, dest_tsv)
        
        # Package ZIP
        zip_path = BASE_DIR / "Brad_Pitt_submission.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            zf.write(dest_tsv, "output/matching_results.tsv")
            cand_tsv = BASE_DIR / "output" / "candidate_pairs.tsv"
            if cand_tsv.exists(): zf.write(cand_tsv, "output/candidate_pairs.tsv")
            for doc in ["Documentation_template.md", "documentation.md"]:
                dp = BASE_DIR / doc
                if dp.exists(): zf.write(dp, doc)
            code_dir = BASE_DIR / "code" / "business_entity_resolution"
            if code_dir.exists():
                for cp in code_dir.rglob("*"):
                    if cp.is_file() and "__pycache__" not in str(cp):
                        zf.write(cp, str(cp.relative_to(BASE_DIR)).replace("\\", "/"))
                        
        shutil.copyfile(zip_path, DESKTOP / "Brad_Pitt_submission.zip")
        print(f"  -> PACKAGED AND READY ON DESKTOP: {DESKTOP / 'Brad_Pitt_submission.zip'}")
        return True
    return False

def push_next_queued():
    if not PENDING_QUEUE:
        print("Queue is empty. All models deployed.")
        return
    next_tag = PENDING_QUEUE.pop(0)
    k_dir = BASE_DIR / f"kaggle_kernel_{next_tag}"
    if k_dir.exists():
        print(f"Pushing next queued model: {next_tag.upper()}...")
        cmd = ["python", "-m", "kaggle", "kernels", "push", "-p", str(k_dir)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(f"  Result: {r.stdout.strip()}")

def main():
    print("=" * 70)
    print("KAGGLE QUEUE MONITOR STARTED (Zero Local CPU)")
    print(f"Monitoring {len(ACTIVE_KERNELS)} active kernels; {len(PENDING_QUEUE)} queued.")
    print("=" * 70)

    completed = set()
    while len(completed) < len(ACTIVE_KERNELS):
        for k in ACTIVE_KERNELS:
            if k in completed: continue
            st = check_status(k)
            print(f"[{time.strftime('%H:%M:%S')}] {k.split('/')[-1]}: {st}")
            if st == "COMPLETE":
                print(f"*** KERNEL COMPLETED: {k} ***")
                download_and_package(k)
                completed.add(k)
                push_next_queued()
            elif st == "ERROR":
                print(f"Kernel error detected for {k}. Freeing slot.")
                completed.add(k)
                push_next_queued()
        time.sleep(60)

if __name__ == "__main__":
    main()
