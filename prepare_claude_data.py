#!/usr/bin/env python3
"""
Prepares the split dataset and prompt package for Claude 3.7 Sonnet Max.
Creates clean, compact TSV samples (<15 MB) containing:
1. train_ground_truth_sample.tsv: 2,000 real training entities with ground truth.
2. train_records_sample.tsv: The exact records for those 2,000 S1 queries and their targets across S2 & S3.
3. test_queries_sample.tsv: 1,000 test entities across India, US, and France.
4. problem_statement_and_diagnostics.md: Full problem statement, leaderboard history, metric analysis, and code snippets.
"""

import sys, os, time
from pathlib import Path

BASE_DIR = Path(r".")
OUT_DIR = BASE_DIR / "claude_share"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = BASE_DIR / "student_resource" / "dataset" / "train"
TEST_DIR = BASE_DIR / "student_resource" / "dataset" / "test"

print("1. Extracting 2,000 GT pairs...")
gt_sample = {}
needed_s1 = set()
needed_tgts = set()

with open(TRAIN_DIR / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
    header = f.readline()
    for i, line in enumerate(f):
        if i >= 2000: break
        p = line.rstrip("\r\n").split("\t")
        sid = p[0]
        tgts = [x.strip() for x in p[1].split(",") if x.strip()] if len(p) > 1 and p[1] else []
        gt_sample[sid] = tgts
        needed_s1.add(sid)
        needed_tgts.update(tgts)

with open(OUT_DIR / "train_ground_truth_sample.tsv", "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for sid, tgts in gt_sample.items():
        f.write(f"{sid}\t{','.join(tgts)}\n")

print(f"  Extracted {len(gt_sample)} GT rows ({len(needed_tgts)} target matches)")

print("2. Extracting S1 training records...")
with open(TRAIN_DIR / "train_source1.tsv", "r", encoding="utf-8") as f_in, \
     open(OUT_DIR / "train_source1_sample.tsv", "w", encoding="utf-8") as f_out:
    header = f_in.readline()
    f_out.write(header)
    for line in f_in:
        p = line.split("\t")
        if p[0] in needed_s1:
            f_out.write(line)

print("3. Extracting target training records (S2 + S3)...")
with open(OUT_DIR / "train_targets_sample.tsv", "w", encoding="utf-8") as f_out:
    f_out.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        with open(TRAIN_DIR / fn, "r", encoding="utf-8") as f_in:
            f_in.readline()
            for line in f_in:
                p = line.split("\t")
                if p[0] in needed_tgts:
                    f_out.write(line)

print("4. Extracting test sample across India, US, and France...")
with open(TEST_DIR / "test_source1.tsv", "r", encoding="utf-8") as f_in, \
     open(OUT_DIR / "test_source1_sample.tsv", "w", encoding="utf-8") as f_out:
    header = f_in.readline()
    f_out.write(header)
    counts = {"India": 0, "US": 0, "France": 0}
    for line in f_in:
        p = line.rstrip("\r\n").split("\t")
        c = p[3].strip() if len(p) > 3 else ""
        if c in counts and counts[c] < 400:
            f_out.write(line)
            counts[c] += 1
        if all(v >= 400 for v in counts.values()):
            break

print(f"Done! Samples written to {OUT_DIR}")
for f in OUT_DIR.glob("*"):
    print(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")
