import json
from pathlib import Path

v11_code = open("run_v11_precision_strike.py", "r", encoding="utf-8").read()

dataset_url = "https://amazon-ml-challenge-dataset.s3.ap-south-1.amazonaws.com/dataset_test.zip"

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Amazon ML Challenge 2026 - Pipeline V11 (Precision Strike Resolver)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["!pip install -q anyascii rapidfuzz"]
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
                "print('Downloading Test Dataset...')\n",
                f"urllib.request.urlretrieve('{dataset_url}', 'dataset_test.zip')\n",
                "with zipfile.ZipFile('dataset_test.zip', 'r') as zf:\n",
                "    zf.extractall('student_resource/dataset/test/')\n",
                "os.remove('dataset_test.zip')\n",
                "print('Dataset ready!')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": v11_code.splitlines(keepends=True)
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

deploy_dir = Path("kaggle_deploy_v11")
deploy_dir.mkdir(exist_ok=True)
with open(deploy_dir / "Kaggle_Amazon_ML_Online.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("V11 package updated with fresh S3 URL!")
