#!/usr/bin/env python3
"""
Multi-Account Kaggle Fleet Manager & Auto-Downloader
Monitors all 14 active cloud kernels across the 3 Kaggle accounts:
- Node-Alpha
- Node-Beta
- Node-Gamma

When any kernel completes:
1. Downloads the output TSV/model
2. Re-packages the final Brad_Pitt_submission.zip on Desktop
3. Pushes V40 into the newly freed slot
Zero CPU usage (runs sleep intervals between lightweight API calls).
"""

import os, sys, time, shutil, subprocess, zipfile
from pathlib import Path

BASE_DIR = Path(r".")
DESKTOP = Path(r"~/Desktop")

ACCOUNTS = {
    'node_alpha': {
        'token': os.environ.get('KAGGLE_API_TOKEN', 'YOUR_KAGGLE_API_TOKEN'),
        'kernels': [
            'amazon-ml-zero-dep-grandmaster',
            'amazon-ml-v26-ultra-precision',
            'amazon-ml-v27-gpu-accelerated',
            'amazon-ml-v28-tri-branch-ranker',
            'amazon-ml-v29-deep-tree'
        ]
    },
    'node_beta': {
        'token': os.environ.get('KAGGLE_API_TOKEN', 'YOUR_KAGGLE_API_TOKEN'),
        'kernels': [
            'amazon-ml-v30-sparse-regularized',
            'amazon-ml-v31-focal-loss',
            'amazon-ml-v32-catboost-native',
            'amazon-ml-v33-contrastive-margin',
            'amazon-ml-v34-rank-xendcg'
        ]
    },
    'node_gamma': {
        'token': os.environ.get('KAGGLE_API_TOKEN', 'YOUR_KAGGLE_API_TOKEN'),
        'kernels': [
            'amazon-ml-v35-asymmetric-f05',
            'amazon-ml-v36-geo-enhanced',
            'amazon-ml-v37-phonetic-dense',
            'amazon-ml-v38-triplet-loss',
            'amazon-ml-v39-super-ensemble'
        ]
    },
    'node_delta': {
        'token': os.environ.get('KAGGLE_API_TOKEN', 'YOUR_KAGGLE_API_TOKEN'),
        'kernels': [
            'amazon-ml-v41-apex-grandmaster',
            'amazon-ml-v42-ngram-tfidf',
            'amazon-ml-v43-geo-hierarchy',
            'amazon-ml-v44-catboost-focal',
            'amazon-ml-v45-bayesian-f05'
        ]
    }
}

V40_DEPLOYED = False

def check_status(user, token, slug):
    os.environ['KAGGLE_API_TOKEN'] = token
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    ref = f"{user}/{slug}"
    try:
        st = api.kernels_status(ref)
        s = str(st.status)
        if "COMPLETE" in s: return "COMPLETE"
        if "RUNNING" in s: return "RUNNING"
        if "QUEUED" in s: return "QUEUED"
        if "ERROR" in s: return "ERROR"
        return s
    except Exception as e:
        return f"ERR_{e}"

def download_and_repackage(user, token, slug):
    global V40_DEPLOYED
    ref = f"{user}/{slug}"
    out_dir = BASE_DIR / f"kaggle_output_{slug}"
    out_dir.mkdir(exist_ok=True)
    print(f"\n[OUTPUT] Downloading artifacts for {ref} to {out_dir.name}...")
    
    os.environ['KAGGLE_API_TOKEN'] = token
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    try:
        api.kernels_output(ref, path=str(out_dir))
    except Exception as e:
        print(f"Download error: {e}")
        return False
        
    # Check if raw_scores file exists to apply closed-form Expected F-0.5 optimization
    raw_scores_file = None
    for p in out_dir.rglob("raw_scores*.tsv"):
        raw_scores_file = p; break

    matching_tsv = None
    if raw_scores_file and raw_scores_file.exists():
        print(f"  [BAYESIAN OPTIMIZER] Applying Expected F-0.5 stopping rule on {raw_scores_file.name}...")
        try:
            from apply_expected_f05_postprocessing import select_matches_f05
            opt_tsv = out_dir / "matching_results_expected_f05.tsv"
            with open(raw_scores_file, "r", encoding="utf-8") as rf, open(opt_tsv, "w", encoding="utf-8") as wf:
                wf.write("source1_entity_id\tmatched_entity_ids\n")
                next(rf) # header
                for line in rf:
                    parts = line.strip().split("\t")
                    qid = parts[0]
                    if len(parts) > 1 and parts[1]:
                        cands, probs = [], []
                        for item in parts[1].split(";"):
                            if ":" in item:
                                cid, pr = item.split(":", 1)
                                cands.append(cid)
                                probs.append(float(pr))
                        chosen = select_matches_f05(cands, probs, beta=0.5)
                        wf.write(f"{qid}\t{','.join(chosen)}\n")
                    else:
                        wf.write(f"{qid}\t\n")
            matching_tsv = opt_tsv
            print(f"  [BAYESIAN OPTIMIZER] Generated closed-form optimal {matching_tsv.name}!")
        except Exception as e:
            print(f"  [BAYESIAN OPTIMIZER] Fallback to raw matching_results: {e}")

    matching_tsv = matching_tsv or None
    if not matching_tsv:
        for p in out_dir.rglob("matching_results*.tsv"):
            matching_tsv = p; break
        
    if matching_tsv and matching_tsv.exists():
        size_mb = matching_tsv.stat().st_size / 1024 / 1024
        print(f"  [SUCCESS] Found predictions: {matching_tsv.name} ({size_mb:.2f} MB)")
        
        # Copy to output and Desktop
        dest_tsv = BASE_DIR / "output" / "matching_results.tsv"
        shutil.copyfile(matching_tsv, dest_tsv)
        desktop_tsv = DESKTOP / "matching_results.tsv"
        shutil.copyfile(matching_tsv, desktop_tsv)
        print(f"  -> DIRECT TSV SYNCED TO DESKTOP: {desktop_tsv}")
        
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
        print(f"  -> UPDATED AND SYNCHRONIZED: {DESKTOP / 'Brad_Pitt_submission.zip'}")
        
        # If V40 not yet deployed, deploy it to this user's newly freed slot!
        if not V40_DEPLOYED:
            deploy_v40(user, token)
            V40_DEPLOYED = True
        return True
    return False

def deploy_v40(user, token):
    print(f"\n[DEPLOY] Slot opened on @{user}! Pushing V40...")
    import json
    meta_path = BASE_DIR / "kaggle_kernel_v40" / "kernel-metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta['id'] = f"{user}/amazon-ml-v40-ultimate-blend"
    meta['title'] = "Amazon ML V40 Ultimate Blend"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
        
    os.environ['KAGGLE_API_TOKEN'] = token
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    try:
        res = api.kernels_push(str(BASE_DIR / "kaggle_kernel_v40"))
        print(f"  V40 successfully pushed: {res.ref}")
    except Exception as e:
        print(f"  V40 push error: {e}")

def main():
    print("=" * 75)
    print("14-KERNEL MULTI-ACCOUNT KAGGLE FLEET MANAGER ACTIVE")
    print("Zero CPU local usage. Monitoring Node-Alpha, Node-Beta, Node-Gamma")
    print("=" * 75)
    
    completed = set()
    total_kernels = sum(len(data['kernels']) for data in ACCOUNTS.values())
    
    while len(completed) < total_kernels:
        print(f"\n[{time.strftime('%H:%M:%S')}] Polling 14 Cloud Kernels...")
        for user, data in ACCOUNTS.items():
            token = data['token']
            for slug in data['kernels']:
                ref = f"{user}/{slug}"
                if ref in completed:
                    continue
                st = check_status(user, token, slug)
                print(f"  {ref:<45} : {st}")
                if "COMPLETE" in st:
                    completed.add(ref)
                    succ = download_and_repackage(user, token, slug)
                    if succ:
                        print(f"\n[ENSEMBLE] Triggering model comparison and consensus ensemble...")
                        try:
                            import compare_and_ensemble
                            compare_and_ensemble.compare_and_build_ensemble()
                        except Exception as ce:
                            print(f"[ENSEMBLE] Error during comparison/ensemble: {ce}")
                elif "ERROR" in st:
                    completed.add(ref)
                    print(f"  [ALERT] {ref} encountered error.")
        
        if len(completed) >= total_kernels:
            print("\nAll kernels processed!")
            break
            
        time.sleep(60)

if __name__ == "__main__":
    main()
