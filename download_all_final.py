import os, sys, io, requests, zipfile, shutil
from pathlib import Path
from kaggle.api import kaggle_api_extended as k

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

api = k.KaggleApi()
api.authenticate()

desktop = Path(r"~/Desktop")
if not desktop.exists():
    desktop = Path(r"~/Desktop")

kernels = [
    ("node_beta", "amazon-ml-challenge-v3-resolver", "v7"),
    ("node_beta", "amazon-ml-challenge-v8-dp-resolver", "v8"),
    ("node_beta", "amazon-ml-challenge-v9-precision-dp", "v9"),
    ("node_beta", "amazon-ml-challenge-v10-phonetic", "v10"),
]

for owner, slug, ver in kernels:
    print("=" * 70)
    print(f"DOWNLOADING COMPLETED ARTIFACTS FOR {slug} ({ver.upper()})...")
    print("=" * 70)
    
    with api.build_kaggle_client() as client:
        req = k.ApiListKernelSessionOutputRequest()
        req.user_name = owner
        req.kernel_slug = slug
        resp = client.kernels.kernels_api_client.list_kernel_session_output(req)
        
        target_dir = Path(f"completed_{ver}")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for item in resp.files or []:
            fname = item.file_name
            # Only download submission files, skip test dataset copies
            if fname in ("submission.zip", "output/matching_results.tsv", "output/candidate_pairs.tsv"):
                out_path = target_dir / fname
                out_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"Downloading {fname}...")
                r = requests.get(item.url, stream=True)
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=2*1024*1024):
                        f.write(chunk)
                print(f"  -> Saved {out_path} ({out_path.stat().st_size:,} bytes)")

        # Copy matching_results.tsv to desktop
        match_tsv = target_dir / "output" / "matching_results.tsv"
        sub_zip = target_dir / "submission.zip"
        
        if match_tsv.exists():
            desktop_tsv = desktop / f"matching_results_{ver}.tsv"
            shutil.copy2(match_tsv, desktop_tsv)
            print(f"Copied to Desktop: {desktop_tsv}")
            
            # Analyze metrics
            with open(match_tsv, "r", encoding="utf-8") as f:
                next(f)
                n_total = 0
                n_empty = 0
                total_matches = 0
                for line in f:
                    n_total += 1
                    p = line.strip().split("\t")
                    if len(p) < 2 or not p[1]:
                        n_empty += 1
                    else:
                        total_matches += len(p[1].split(","))
                        
            print(f"\n[{ver.upper()} METRICS]")
            print(f"  Total Queries:     {n_total:,}")
            print(f"  Singletons (Empty):{n_empty:,} ({n_empty/n_total*100:.2f}%)  [Ground Truth: 5.58%]")
            print(f"  With Matches:      {n_total-n_empty:,} ({(n_total-n_empty)/n_total*100:.2f}%)")
            print(f"  Total Match Pairs: {total_matches:,}")
            print(f"  Avg Matches/Entity:{total_matches/max(1, n_total-n_empty):.2f}  [Ground Truth: 3.46]")
            
        if sub_zip.exists():
            desktop_zip = desktop / f"submission_{ver}.zip"
            shutil.copy2(sub_zip, desktop_zip)
            print(f"Copied to Desktop: {desktop_zip}")
            
    print()

# Also set the best (V8) as the primary matching_results.tsv and submission.zip on Desktop
v8_tsv = desktop / "matching_results_v8.tsv"
v8_zip = desktop / "submission_v8.zip"
if v8_tsv.exists():
    shutil.copy2(v8_tsv, desktop / "matching_results.tsv")
    print("SET PRIMARY: Desktop/matching_results.tsv -> V8 (Grandmaster Expected F_0.5 DP)")
if v8_zip.exists():
    shutil.copy2(v8_zip, desktop / "submission.zip")
    print("SET PRIMARY: Desktop/submission.zip -> V8")
