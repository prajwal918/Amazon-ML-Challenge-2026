#!/usr/bin/env python3
"""
Automates generation and Kaggle push for V30 through V40.
"""

import os, sys, shutil, subprocess
from pathlib import Path

BASE_DIR = Path(r".")
TEMPLATE_PY = BASE_DIR / "kaggle_v29_deep_tree.py"

MODELS = [
    ("v30", "amazon-ml-v30-sparse-regularized", "Amazon ML V30 Sparse Regularized", 127, 8, 0.02, 2.0, 8.0, 0.25, 0.85, 0.60),
    ("v31", "amazon-ml-v31-precision-92", "Amazon ML V31 Precision 92", 63, 8, 0.03, 0.2, 2.0, 0.18, 0.92, 0.72),
    ("v32", "amazon-ml-v32-bagging-ensemble", "Amazon ML V32 Bagging Ensemble", 63, 7, 0.03, 0.5, 3.0, 0.22, 0.86, 0.62),
    ("v33", "amazon-ml-v33-phonetic-boost", "Amazon ML V33 Phonetic Boost", 95, 8, 0.025, 0.3, 2.0, 0.25, 0.86, 0.62),
    ("v34", "amazon-ml-v34-address-anchor", "Amazon ML V34 Address Anchor", 63, 8, 0.03, 0.2, 2.0, 0.25, 0.84, 0.58),
    ("v35", "amazon-ml-v35-asymmetric-f05", "Amazon ML V35 Asymmetric F05", 63, 8, 0.03, 0.2, 2.0, 0.15, 0.89, 0.66),
    ("v36", "amazon-ml-v36-jaro-heavy", "Amazon ML V36 Jaro Heavy", 79, 8, 0.025, 0.3, 2.0, 0.22, 0.87, 0.64),
    ("v37", "amazon-ml-v37-ngram-overlap", "Amazon ML V37 Ngram Overlap", 95, 8, 0.025, 0.2, 2.0, 0.24, 0.85, 0.60),
    ("v38", "amazon-ml-v38-threshold-fleet", "Amazon ML V38 Threshold Fleet", 63, 8, 0.03, 0.2, 2.0, 0.25, 0.88, 0.65),
    ("v39", "amazon-ml-v39-greedy-top1", "Amazon ML V39 Greedy Top1", 63, 8, 0.03, 0.2, 2.0, 0.20, 0.90, 0.75),
    ("v40", "amazon-ml-v40-master-blend", "Amazon ML V40 Master Blend", 127, 9, 0.02, 0.5, 3.0, 0.22, 0.86, 0.62),
]

with open(TEMPLATE_PY, "r", encoding="utf-8") as f:
    template_code = f.read()

for tag, slug, title, leaves, depth, lr, alpha, lmbda, spw_ratio, base_th, sib_th in MODELS:
    print(f"Deploying {tag.upper()} ({slug})...")
    k_dir = BASE_DIR / f"kaggle_kernel_{tag}"
    k_dir.mkdir(exist_ok=True)

    # Customize script parameters
    code = template_code.replace("V29 DEEP-TREE COMPLEX INTERACTION RESOLVER", f"{tag.upper()} KAGGLE GRANDMASTER RESOLVER")
    code = code.replace("num_leaves=127, max_depth=10", f"num_leaves={leaves}, max_depth={depth}")
    code = code.replace("learning_rate=0.02", f"learning_rate={lr}")
    code = code.replace("reg_alpha=0.3, reg_lambda=2.0", f"reg_alpha={alpha}, reg_lambda={lmbda}")
    code = code.replace("ratio * 0.25", f"ratio * {spw_ratio}")
    code = code.replace("best_cfg = (0.85, 0.60)", f"best_cfg = ({base_th}, {sib_th})")

    script_name = f"kaggle_{tag}.py"
    with open(k_dir / script_name, "w", encoding="utf-8") as f:
        f.write(code)

    # Write kernel metadata
    meta = f'''{{
  "id": "node_alpha/{slug}",
  "title": "{title}",
  "code_file": "{script_name}",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": ["node_alpha/amazon-ml-challenge-2026"],
  "competition_sources": [],
  "kernel_sources": []
}}
'''
    with open(k_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        f.write(meta)

    # Push to Kaggle
    cmd = ["python", "-m", "kaggle", "kernels", "push", "-p", str(k_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if "successfully pushed" in res.stdout:
        print(f"  -> {tag.upper()} PUSHED successfully!")
    else:
        print(f"  -> {tag.upper()}: {res.stdout.strip()} {res.stderr.strip()[:100]}")

print("\nAll models processed.")
