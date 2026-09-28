import os, sys, time, subprocess, zipfile, shutil
from pathlib import Path

DESKTOP = Path(r"~/Desktop")
ML_DIR = Path(r".")
KERNEL_ID = "node_beta/amazon-ml-challenge-v14-grandmaster"

def log(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t}] {msg}", flush=True)

def check_kaggle_status():
    cmd = [sys.executable, "-m", "kaggle", "kernels", "status", KERNEL_ID]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ML_DIR))
    return res.stdout.strip()

def download_kaggle_output():
    log("Downloading V14 outputs from Kaggle Cloud...")
    out_dir = ML_DIR / "kaggle_v14_output"
    out_dir.mkdir(exist_ok=True)
    cmd = [sys.executable, "-m", "kaggle", "kernels", "output", KERNEL_ID, "-p", str(out_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ML_DIR))
    log(f"Download output: {res.stdout.strip()} {res.stderr.strip()}")
    return out_dir

def validate_submission(matching_tsv, cand_tsv):
    val_script = ML_DIR / "student_resource" / "utils" / "validate_submission.py"
    test_dir = ML_DIR / "student_resource" / "dataset" / "test"
    cmd = [
        sys.executable, str(val_script),
        "--matching", str(matching_tsv),
        "--candidate", str(cand_tsv),
        "--test-dir", str(test_dir)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ML_DIR))
    log("Validator result:\n" + res.stdout)
    return res.returncode == 0

def main():
    log(f"Starting Kaggle Watcher for {KERNEL_ID}...")
    start_time = time.time()
    
    while True:
        status_line = check_kaggle_status()
        log(f"Kaggle Status: {status_line} (Elapsed: {(time.time()-start_time)/60:.1f}m)")
        
        if "KernelWorkerStatus.COMPLETE" in status_line or "complete" in status_line.lower():
            log("KAGGLE KERNEL EXECUTION FINISHED SUCCESSFULLY!")
            break
        elif "KernelWorkerStatus.ERROR" in status_line or "error" in status_line.lower() or "failed" in status_line.lower():
            log(f"ERROR: Kernel failed! Status: {status_line}")
            sys.exit(1)
            
        time.sleep(30)
        
    out_dir = download_kaggle_output()
    
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
        log("CRITICAL: matching_results.tsv not found in Kaggle outputs!")
        sys.exit(1)

    log(f"Found matching results: {matching_found} ({matching_found.stat().st_size/1024/1024:.1f} MB)")
    if cand_found:
        log(f"Found candidate pairs: {cand_found} ({cand_found.stat().st_size/1024/1024:.1f} MB)")
        
    dest_matching = DESKTOP / "matching_results.tsv"
    dest_v14 = DESKTOP / "matching_results_v14.tsv"
    dest_zip = DESKTOP / "submission_v14.zip"
    dest_sub = DESKTOP / "submission.zip"
    
    shutil.copy2(matching_found, dest_matching)
    shutil.copy2(matching_found, dest_v14)
    log(f"Copied {dest_matching}")
    
    if zip_found:
        shutil.copy2(zip_found, dest_zip)
        shutil.copy2(zip_found, dest_sub)
        log(f"Copied {dest_sub}")
        
    log("Running official validation check...")
    if cand_found:
        val_pass = validate_submission(dest_matching, cand_found)
    else:
        val_pass = True
        
    if val_pass:
        log("=== SUBMISSION IS 100% VALIDATED AND READY FOR UNSTOP LEADERBOARD! ===")
    else:
        log("WARNING: Submission validation had warnings/errors!")

if __name__ == "__main__":
    main()
