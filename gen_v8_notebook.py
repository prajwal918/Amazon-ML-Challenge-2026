import json, os
from pathlib import Path

v8_code = open("run_v8_dp_resolver.py", "r", encoding="utf-8").read()

dataset_url = "https://amazon-ml-challenge-dataset.s3.ap-south-1.amazonaws.com/dataset_test.zip"
model_url = "https://amazon-ml-challenge-dataset.s3.ap-south-1.amazonaws.com/lgb_entity_resolver.txt"

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Amazon ML Challenge 2026 - Pipeline V8 (Grandmaster DP Resolver)\n",
                "- Pre-Vectorization Normalization (De-leetspeak, Accents, Legal Suffixes)\n",
                "- 5-Key Inverted Index (Tokens, Addr+Num, Addr-Only, Prefix 4-gram)\n",
                "- LightGBM Pairwise Probability Model (SIMD RapidFuzz features)\n",
                "- Faron / Ye et al. (ICML 2012) Expected F_0.5 Dynamic Programming per query"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q anyascii rapidfuzz lightgbm"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, urllib.request, zipfile\n",
                "os.makedirs('student_resource/dataset/test', exist_ok=True)\n",
                "os.makedirs('student_resource/models', exist_ok=True)\n",
                "os.makedirs('output', exist_ok=True)\n\n",
                "print('Downloading LightGBM model...')\n",
                f"urllib.request.urlretrieve('{model_url}', 'student_resource/models/lgb_entity_resolver.txt')\n",
                "print('Model downloaded:', os.path.getsize('student_resource/models/lgb_entity_resolver.txt'), 'bytes')\n\n",
                "print('Downloading test dataset...')\n",
                f"urllib.request.urlretrieve('{dataset_url}', 'dataset_test.zip')\n",
                "with zipfile.ZipFile('dataset_test.zip', 'r') as zf:\n",
                "    zf.extractall('student_resource/dataset/test/')\n",
                "os.remove('dataset_test.zip')\n",
                "print('Test dataset ready:', os.listdir('student_resource/dataset/test'))"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": v8_code.splitlines(keepends=True)
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!ls -lh output/\n",
                "!wc -l output/matching_results.tsv\n",
                "!head -5 output/matching_results.tsv"
            ]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"},
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

deploy_dir = Path("kaggle_deploy_v8")
deploy_dir.mkdir(exist_ok=True)
nb_path = deploy_dir / "Kaggle_Amazon_ML_Online.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

meta = {
    "id": "node_beta/amazon-ml-challenge-v8-dp",
    "title": "Amazon ML Challenge V8 DP Resolver",
    "code_file": "Kaggle_Amazon_ML_Online.ipynb",
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
with open(deploy_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("V8 deployment package ready at:", deploy_dir)
