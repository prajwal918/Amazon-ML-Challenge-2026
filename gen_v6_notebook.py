import json
from pathlib import Path

v6_code = open("run_v6_max_recall.py", "r", encoding="utf-8").read()

dataset_url = "https://amazon-ml-challenge-dataset.s3.ap-south-1.amazonaws.com/dataset_test.zip"

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Amazon ML Challenge 2026 - V6 Maximum Recall Pipeline\n",
                "5 blocking keys, address-only blocking, name prefix ngrams, partial_ratio, higher posting caps"
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
                "print('Downloading test dataset from S3...')\n",
                f"url = \"{dataset_url}\"\n",
                "urllib.request.urlretrieve(url, 'dataset_test.zip')\n",
                "print('Downloaded! Extracting...')\n",
                "with zipfile.ZipFile('dataset_test.zip', 'r') as zf:\n",
                "    zf.extractall('student_resource/dataset/test/')\n",
                "print('Test data ready:', os.listdir('student_resource/dataset/test'))\n",
                "os.remove('dataset_test.zip')\n",
                "print('Cleaned up zip')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": v6_code.splitlines(keepends=True)
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

nb_path = Path("kaggle_deploy/Kaggle_Amazon_ML_Online.ipynb")
nb_path.parent.mkdir(exist_ok=True)
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("V6 Notebook generated:", nb_path)
print("URL length:", len(dataset_url))
