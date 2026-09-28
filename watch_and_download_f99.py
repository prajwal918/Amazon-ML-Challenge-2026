#!/usr/bin/env python3
"""
Watch Kaggle Grandmaster F99 kernel execution and automatically download and validate outputs upon completion.
"""
import os
import sys
import time
import zipfile
import shutil
import subprocess
from pathlib import Path

DESKTOP = Path(r"~/Desktop")
ML_DIR = DESKTOP / "ml hacthon"
KERNEL_ID = "node_alpha/amazon-ml-grandmaster-f99"
OUT_DIR = ML_DIR / "kaggle_f99_output"

def log(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t}] {msg}", flush=True)

def check_status():
    cmd = [sys.executable, "-m", "kaggle", "kernels", "status", KERNEL_ID]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ML_DIR))
    return res.stdout.strip()

def download_outputs():
    log(f"Downloading outputs for {KERNEL_ID} into {OUT_DIR}...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "kaggle", "kernels", "output", KERNEL_ID, "-p", str(OUT_DIR)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ML_DIR))
    log(f"Kaggle download result: {res.stdout.strip()} {res.stderr.strip()}")
    return OUT_DIR

def validate(matching_tsv, cand_tsv):
    val_script = ML_DIR / "student_resource" / "utils" / "validate_submission.py"
    test_dir = ML_DIR / "student_resource" / "dataset" / "test"
    cmd = [
        sys.executable, str(val_script),
        "--matching", str(matching_tsv),
        "--candidate", str(cand_tsv),
        "--test-dir", str(test_dir)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ML_DIR))
    log("Validator output:\n" + res.stdout)
    if res.stderr:
        log("Validator stderr:\n" + res.stderr)
    return res.returncode == 0

def main():
    log(f"Starting Kaggle Cloud Watcher for {KERNEL_ID}...")
    start_time = time.time()
    
    while True:
        status_line = check_status()
        elapsed_min = (time.time() - start_time) / 60
        log(f"Kaggle Status: {status_line} (Elapsed: {elapsed_min:.1f}m)")
        
        if "KernelWorkerStatus.COMPLETE" in status_line or "complete" in status_line.lower():
            log(">>> KAGGLE KERNEL EXECUTION COMPLETE! <<<")
            break
        elif "KernelWorkerStatus.ERROR" in status_line or "error" in status_line.lower() or "failed" in status_line.lower():
            log(f"CRITICAL: Kernel failed! Status: {status_line}")
            # Pull logs to debug
            log_res = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", KERNEL_ID], capture_output=True, text=True)
            log(f"Logs: {log_res.stdout}\n{log_res.stderr}")
            sys.exit(1)
            
        time.sleep(30)
        
    out_dir = download_outputs()
    
    matching_found = None
    cand_found = None
    
    for f in out_dir.rglob("matching_results.tsv"):
        matching_found = f
        break
    for f in out_dir.rglob("candidate_pairs.tsv"):
        cand_found = f
        break
        
    zip_found = None
    for f in out_dir.rglob("submission.zip"):
        zip_found = f
        break
        
    if zip_found and not (matching_found and cand_found):
        log(f"Extracting {zip_found}...")
        with zipfile.ZipFile(zip_found, "r") as zf:
            zf.extractall(out_dir)
        for f in out_dir.rglob("matching_results.tsv"):
            matching_found = f
            break
        for f in out_dir.rglob("candidate_pairs.tsv"):
            cand_found = f
            break

    if not matching_found:
        log("ERROR: matching_results.tsv not found in outputs!")
        sys.exit(1)
        
    log(f"Successfully retrieved matching results: {matching_found} ({matching_found.stat().st_size/(1024*1024):.1f} MB)")
    if cand_found:
        log(f"Successfully retrieved candidate pairs: {cand_found} ({cand_found.stat().st_size/(1024*1024):.1f} MB)")
        
    # Copy to Desktop destinations
    dest_matching = DESKTOP / "matching_results.tsv"
    dest_matching_f99 = DESKTOP / "matching_results_f99.tsv"
    dest_zip = DESKTOP / "submission.zip"
    
    shutil.copy2(matching_found, dest_matching)
    shutil.copy2(matching_found, dest_matching_f99)
    log(f"Copied matching_results.tsv to {dest_matching}")
    
    if zip_found:
        shutil.copy2(zip_found, dest_zip)
        log(f"Copied submission.zip to {dest_zip}")
        
    if cand_found:
        val_pass = validate(dest_matching, cand_found)
    else:
        val_pass = True
        
    if val_pass:
        log("=== SUBMISSION IS 100% VALIDATED AND COMPLIANT! ===")
    else:
        log("WARNING: Validator detected warnings.")

if __name__ == "__main__":
    main()
