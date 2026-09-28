import sys, os, io, shutil, subprocess
from pathlib import Path

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Monkey-patch open for UTF-8 in Kaggle API
original_open = open
def utf8_open(*args, **kwargs):
    if len(args) >= 2 and 'w' in str(args[1]) and 'b' not in str(args[1]):
        kwargs.setdefault('encoding', 'utf-8')
    return original_open(*args, **kwargs)

import builtins
builtins.open = utf8_open

import kaggle

def process_kernel(kernel_slug, version_name):
    print("=" * 70)
    print(f"Checking {kernel_slug} ({version_name})...")
    print("=" * 70)
    
    api = kaggle.KaggleApi()
    api.authenticate()
    
    status_resp = api.kernels_status(kernel_slug)
    status = status_resp.status
    str_status = str(status).lower()
    print(f"Status: {status} (str: {str_status})")
    
    out_dir = Path(f"kaggle_output_{version_name.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if "complete" in str_status:
        print(f"KERNEL COMPLETE! Downloading artifacts to {out_dir}...")
        api.kernels_output(kernel_slug, str(out_dir))
        print("Downloaded files:", os.listdir(out_dir))
        
        match_tsv = out_dir / "matching_results.tsv"
        cand_tsv = out_dir / "candidate_pairs.tsv"
        sub_zip = out_dir / "submission.zip"
        
        # If output was inside an 'output' subfolder
        if (out_dir / "output" / "matching_results.tsv").exists():
            match_tsv = out_dir / "output" / "matching_results.tsv"
            cand_tsv = out_dir / "output" / "candidate_pairs.tsv"
            
        desktop_dir = Path(r"~/Desktop")
        if not desktop_dir.exists():
            desktop_dir = Path(r"~/Desktop")
            
        if match_tsv.exists():
            dest_tsv = desktop_dir / f"matching_results_{version_name.lower()}.tsv"
            shutil.copy2(match_tsv, dest_tsv)
            # Also copy as main matching_results.tsv for easy upload
            shutil.copy2(match_tsv, desktop_dir / "matching_results.tsv")
            print(f"COPIED to Desktop: {dest_tsv} and matching_results.tsv")
            
            # Compute stats
            with open(match_tsv, "r", encoding="utf-8") as f:
                header = next(f)
                n_total = 0
                n_empty = 0
                total_matches = 0
                for line in f:
                    n_total += 1
                    parts = line.strip().split("\t")
                    if len(parts) < 2 or not parts[1]:
                        n_empty += 1
                    else:
                        total_matches += len(parts[1].split(","))
                        
            print(f"\n--- SUBMISSION METRICS ---")
            print(f"  Total Queries:      {n_total:,}")
            print(f"  Singletons:         {n_empty:,} ({n_empty/n_total*100:.2f}%)")
            print(f"  With Matches:       {n_total-n_empty:,} ({(n_total-n_empty)/n_total*100:.2f}%)")
            print(f"  Total Match Pairs:  {total_matches:,}")
            print(f"  Avg Matches/Entity: {total_matches/max(1, n_total-n_empty):.2f}")
            
            # Validate
            print(f"\n--- RUNNING OFFICIAL VALIDATOR ---")
            cmd = [
                sys.executable,
                "student_resource/utils/validate_submission.py",
                "--matching", str(match_tsv),
                "--test-dir", "student_resource/dataset/test"
            ]
            if cand_tsv.exists():
                cmd.extend(["--candidate", str(cand_tsv)])
            res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            print(res.stdout)
            if res.stderr:
                print("Validator stderr:", res.stderr)
                
            return True
            
        elif sub_zip.exists():
            print("Found submission.zip, extracting...")
            import zipfile
            with zipfile.ZipFile(sub_zip, "r") as zf:
                zf.extractall(out_dir)
            return process_kernel(kernel_slug, version_name)
            
    elif "error" in str_status:
        print(f"ERROR: Kernel failed! Downloading log...")
        api.kernels_output(kernel_slug, str(out_dir))
        for f in os.listdir(out_dir):
            if f.endswith(".log"):
                with open(out_dir / f, "r", encoding="utf-8", errors="replace") as lf:
                    lines = lf.readlines()
                    print("Log tail:")
                    print("".join(lines[-25:]))
    else:
        print(f"Still running... (Status: {status})")
        
    return False

if __name__ == "__main__":
    slug = sys.argv[1] if len(sys.argv) > 1 else "node_beta/amazon-ml-challenge-v3-resolver"
    ver = sys.argv[2] if len(sys.argv) > 2 else "v7"
    process_kernel(slug, ver)
