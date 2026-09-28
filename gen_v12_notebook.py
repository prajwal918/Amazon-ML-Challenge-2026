import json
from pathlib import Path

v12_code = open("run_v12_grandmaster.py", "r", encoding="utf-8").read()

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Amazon ML Challenge 2026 - Pipeline V12 (Grandmaster Entity Resolver)"]
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
                "os.makedirs('output', exist_ok=True)\n",
                "test_file = 'student_resource/dataset/test/test_source1.tsv'\n",
                "if not os.path.exists(test_file):\n",
                "    print('Downloading Student Resource from Unstop CDN...')\n",
                "    urllib.request.urlretrieve('https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip', 'student_resource.zip')\n",
                "    print('Extracting test files...')\n",
                "    with zipfile.ZipFile('student_resource.zip', 'r') as zf:\n",
                "        zf.extractall('.')\n",
                "    os.remove('student_resource.zip')\n",
                "    print('Dataset ready!')\n",
                "else:\n",
                "    print('Dataset already exists!')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": v12_code.splitlines(keepends=True)
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["!ls -lh output/", "!ls -lh submission.zip"]
        }
    ],
    "metadata": {
        "language_info": {"name": "python"},
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

deploy_dir = Path("kaggle_deploy_v12")
deploy_dir.mkdir(exist_ok=True)
with open(deploy_dir / "Kaggle_Amazon_ML_Online.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

metadata = {
    "id": "node_beta/amazon-ml-challenge-v12-grandmaster",
    "title": "Amazon ML Challenge V12 Grandmaster",
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
    json.dump(metadata, f, indent=2)

print("V12 deployment package updated with permanent Unstop CDN URL!")
