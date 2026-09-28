#!/usr/bin/env python3
"""
Deploy and monitor Grandmaster F99 pipeline on Kaggle Cloud.
"""
import os
import sys
import json
import base64
import time
import subprocess
from pathlib import Path

BASE_DIR = Path(r".")
DEPLOY_DIR = BASE_DIR / "kaggle_deploy_grandmaster_f99"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

# 1. Read base64 model
with open(BASE_DIR / "model_f99.pkl", "rb") as f:
    model_b64 = base64.b64encode(f.read()).decode("ascii")

# 2. Read pipeline code
with open(BASE_DIR / "kaggle_pipeline_f99.py", "r", encoding="utf-8") as f:
    pipeline_code = f.read()

# 3. Build Jupyter Notebook cells
cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Amazon ML Challenge 2026 -- Grandmaster Production Pipeline (Target: 0.99+ F0.5)\n",
            "This notebook runs end-to-end multi-source entity resolution across 1.73M queries and 10M target records.\n",
            "Generates both matching_results.tsv and candidate_pairs.tsv."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install required high-performance libraries\n",
            "!pip install -q anyascii rapidfuzz indic-transliteration unidecode lightgbm joblib\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, urllib.request, zipfile, sys\n",
            "from pathlib import Path\n",
            "\n",
            "# Download student resource dataset from Unstop CDN if missing\n",
            "test_file = Path('student_resource/dataset/test/test_source1.tsv')\n",
            "if not test_file.exists():\n",
            "    print('Downloading Student Resource (1.0 GB) from Unstop CDN...')\n",
            "    url = 'https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip'\n",
            "    urllib.request.urlretrieve(url, 'student_resource.zip')\n",
            "    print('Extracting dataset...')\n",
            "    with zipfile.ZipFile('student_resource.zip', 'r') as zf:\n",
            "        zf.extractall('.')\n",
            "    if os.path.exists('student_resource.zip'):\n",
            "        os.remove('student_resource.zip')\n",
            "    print('Dataset extracted successfully!')\n",
            "else:\n",
            "    print('Dataset already present!')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Unpack calibrated LightGBM model\n",
            "import base64\n",
            "MODEL_B64 = '''" + model_b64 + "'''\n",
            "with open('model_f99.pkl', 'wb') as f:\n",
            "    f.write(base64.b64decode(MODEL_B64))\n",
            "print('Calibrated model model_f99.pkl unpacked! Size:', os.path.getsize('model_f99.pkl'), 'bytes')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Write pipeline script\n",
            "with open('pipeline_runner.py', 'w', encoding='utf-8') as f:\n",
            "    f.write('''" + pipeline_code.replace("\\", "\\\\").replace("'''", "\\'\\'\\'") + "''')\n",
            "print('pipeline_runner.py written successfully!')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Execute full Grandmaster Inference\n",
            "import subprocess\n",
            "from pathlib import Path\n",
            "\n",
            "# Locate test files\n",
            "s1_files = list(Path('.').rglob('test_source1.tsv'))\n",
            "s2_files = list(Path('.').rglob('test_source2.tsv'))\n",
            "s3_files = list(Path('.').rglob('test_source3.tsv'))\n",
            "\n",
            "if not s1_files or not s2_files or not s3_files:\n",
            "    raise RuntimeError(f'Missing test files: s1={s1_files}, s2={s2_files}, s3={s3_files}')\n",
            "\n",
            "s1_path = str(s1_files[0])\n",
            "targets_arg = f'{s2_files[0]},{s3_files[0]}'\n",
            "\n",
            "print(f'Source 1: {s1_path}')\n",
            "print(f'Targets: {targets_arg}')\n",
            "\n",
            "cmd = [\n",
            "    'python', 'pipeline_runner.py', 'predict',\n",
            "    '--source1', s1_path,\n",
            "    '--targets', targets_arg,\n",
            "    '--model', 'model_f99.pkl',\n",
            "    '--output-dir', 'output',\n",
            "    '--max-candidates', '25'\n",
            "]\n",
            "print('Running command:', ' '.join(cmd))\n",
            "subprocess.run(cmd, check=True)\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Validate submission files\n",
            "import subprocess\n",
            "from pathlib import Path\n",
            "\n",
            "val_scripts = list(Path('.').rglob('validate_submission.py'))\n",
            "test_dirs = list(Path('.').rglob('dataset/test'))\n",
            "\n",
            "if val_scripts and test_dirs:\n",
            "    val_cmd = [\n",
            "        'python', str(val_scripts[0]),\n",
            "        '--matching', 'output/matching_results.tsv',\n",
            "        '--candidate', 'output/candidate_pairs.tsv',\n",
            "        '--test-dir', str(test_dirs[0])\n",
            "    ]\n",
            "    print('Running submission validation...')\n",
            "    res = subprocess.run(val_cmd, capture_output=True, text=True)\n",
            "    print(res.stdout)\n",
            "    print(res.stderr)\n",
            "else:\n",
            "    print('Validation script or test dir not found, skipping local validation.')\n",
            "\n",
            "!ls -lh output/\n"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python"},
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

nb_path = DEPLOY_DIR / "kaggle_grandmaster_f99.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

metadata = {
    "id": "node_alpha/amazon-ml-grandmaster-f99",
    "title": "Amazon ML Grandmaster F99",
    "code_file": "kaggle_grandmaster_f99.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": []
}

with open(DEPLOY_DIR / "kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"Kaggle deployment notebook and metadata created at {DEPLOY_DIR}")
