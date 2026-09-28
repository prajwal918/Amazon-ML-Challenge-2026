import json
from pathlib import Path

v5_code = open("run_v5_combined.py", "r", encoding="utf-8").read()

# Get the presigned URLs (use the ones we already have)
dataset_url = "https://amazon-ml-challenge-dataset.s3.ap-south-1.amazonaws.com/dataset_test.zip"

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Amazon ML Challenge 2026 - V5 RapidFuzz Hybrid Resolver\n",
                "Two stages: (A) Validate on training data first, (B) Run on test data\n",
                "- RapidFuzz token_sort_ratio for word permutation handling\n",
                "- AnyAscii phonetic romanization for cross-script matching\n",
                "- No candidate cap - scores ALL blocker candidates\n",
                "- Calibrated precision thresholds"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q anyascii rapidfuzz"
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
                "os.makedirs('output', exist_ok=True)\n\n",
                "# Download test dataset\n",
                "print('Downloading test dataset from S3...')\n",
                f"url = '{dataset_url}'\n",
                "urllib.request.urlretrieve(url, 'dataset_test.zip')\n",
                "with zipfile.ZipFile('dataset_test.zip', 'r') as zf:\n",
                "    zf.extractall('student_resource/dataset/test/')\n",
                "print('Test data ready:', os.listdir('student_resource/dataset/test'))"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Write and run the V5 combined pipeline (validation skipped - no training data on Kaggle)\n",
                "with open('run_v5_combined.py', 'w', encoding='utf-8') as f:\n",
                f"    f.write({repr(v5_code)})\n\n",
                "!python3 -u run_v5_combined.py"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!ls -lh output/"
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

nb_path = Path("kaggle_deploy/Kaggle_Amazon_ML_Online.ipynb")
nb_path.parent.mkdir(exist_ok=True)
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("Notebook generated:", nb_path)
