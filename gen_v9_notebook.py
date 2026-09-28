import json
from pathlib import Path

v9_code = open("run_v9_precision_dp.py", "r", encoding="utf-8").read()

dataset_url = "https://amazon-ml-challenge-dataset.s3.ap-south-1.amazonaws.com/dataset_test.zip"
model_url = "https://amazon-ml-challenge-dataset.s3.ap-south-1.amazonaws.com/lgb_entity_resolver.txt"

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Amazon ML Challenge 2026 - Pipeline V9 (Ultra-Precision Faron DP)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["!pip install -q anyascii rapidfuzz lightgbm"]
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
                "print('Downloading Model...')\n",
                f"urllib.request.urlretrieve('{model_url}', 'student_resource/models/lgb_entity_resolver.txt')\n\n",
                "print('Downloading Dataset...')\n",
                f"urllib.request.urlretrieve('{dataset_url}', 'dataset_test.zip')\n",
                "with zipfile.ZipFile('dataset_test.zip', 'r') as zf:\n",
                "    zf.extractall('student_resource/dataset/test/')\n",
                "os.remove('dataset_test.zip')\n",
                "print('Ready!')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": v9_code.splitlines(keepends=True)
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["!ls -lh output/"]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"},
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

deploy_dir = Path("kaggle_deploy_v9")
deploy_dir.mkdir(exist_ok=True)
with open(deploy_dir / "Kaggle_Amazon_ML_Online.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

meta = {
    "id": "node_beta/amazon-ml-challenge-v9-precision-dp",
    "title": "Amazon ML Challenge V9 Precision DP",
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

print("V9 package created at:", deploy_dir)
