#!/usr/bin/env python3
"""
Recall Ceiling & Blocking Evaluation Tool for Amazon ML Challenge 2026.
Measures what fraction of true ground-truth matches are surfaced by blocking.
"""

import sys
import re
import time
import math
from pathlib import Path
from collections import defaultdict
from entity_resolution_metric import evaluate_predictions

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "student_resource" / "dataset" / "train"

LEGAL_SUFFIXES = {
    "corp", "corporation", "pvt", "private", "ltd", "limited",
    "llc", "inc", "incorporated", "co", "company", "sarl", "sas", "llp", "sa", "gmbh",
}
STOPWORDS = {"the", "and", "of", "a", "an", "in", "to", "for", "at", "on", "by"}
ADDRESS_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "apt": "apartment", "hwy": "highway",
}


def normalize_text(text: str) -> str:
    text = str(text).lower().replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_name_tokens(name: str) -> frozenset:
    toks = normalize_text(name).split()
    return frozenset(t for t in toks if t not in LEGAL_SUFFIXES and t not in STOPWORDS and len(t) > 1)


def extract_addr_tokens(addr: str) -> frozenset:
    toks = [ADDRESS_ABBREV.get(t, t) for t in normalize_text(addr).split()]
    return frozenset(t for t in toks if t not in STOPWORDS and len(t) > 1)


def main():
    print("=" * 65)
    print("Amazon ML Challenge 2026 - Blocking Recall Diagnostic")
    print("=" * 65)

    # 1. Load validation slice of ground truth (5,000 non-empty S1 entities)
    print("Sampling 5,000 S1 entities with ground truth...")
    val_gt = {}
    val_s1_ids = set()
    all_target_ground_truth_ids = set()

    with open(TRAIN_DIR / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            s1_id = parts[0]
            matches = set(parts[1].split(",")) if len(parts) > 1 and parts[1].strip() else set()
            val_gt[s1_id] = matches
            val_s1_ids.add(s1_id)
            all_target_ground_truth_ids.update(matches)
            if len(val_gt) >= 5000:
                break

    print(f"Sampled {len(val_gt):,} S1 entities ({sum(1 for m in val_gt.values() if m):,} with true matches)")
    print(f"Total target matches to retrieve: {len(all_target_ground_truth_ids):,}")

    # Load S1 record data for these 5,000 entities
    val_s1_records = {}
    with open(TRAIN_DIR / "train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0] in val_s1_ids:
                val_s1_records[parts[0]] = {
                    "name": parts[1] if len(parts) > 1 else "",
                    "addr": parts[2] if len(parts) > 2 else "",
                    "country": parts[3] if len(parts) > 3 else "",
                }
                if len(val_s1_records) >= len(val_s1_ids):
                    break

    # 2. Index Target entities (from train_source2 and train_source3)
    # We index 1,000,000 rows from each + any rows that appear in all_target_ground_truth_ids
    print("\nIndexing training targets (S2 + S3)...")
    vocab = {}
    def get_tid(token: str) -> int: return vocab.setdefault(token, len(vocab))

    target_ids = []
    target_name_toks = []
    target_addr_toks = []
    name_inverted_index = defaultdict(list)
    addr_inverted_index = defaultdict(list)

    for label, fn in [("S2", "train_source2.tsv"), ("S3", "train_source3.tsv")]:
        t0 = time.time()
        count = 0
        with open(TRAIN_DIR / fn, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                if not parts:
                    continue
                eid = parts[0]
                # Include first 500k rows OR if this eid is in our ground truth target set
                if count < 500000 or eid in all_target_ground_truth_ids:
                    count += 1
                    name = parts[1] if len(parts) > 1 else ""
                    addr = parts[2] if len(parts) > 2 else ""

                    n_toks = extract_name_tokens(name)
                    a_toks = extract_addr_tokens(addr)

                    n_tids = frozenset(get_tid(t) for t in n_toks)
                    a_tids = frozenset(get_tid(t) for t in a_toks)

                    c_idx = len(target_ids)
                    target_ids.append(eid)
                    target_name_toks.append(n_tids)
                    target_addr_toks.append(a_tids)

                    for tid in n_tids:
                        name_inverted_index[tid].append(c_idx)
                    for tid in a_tids:
                        addr_inverted_index[tid].append(c_idx)

        print(f"Indexed {count:,} {label} records in {time.time()-t0:.1f}s")

    # 3. Test Dual-Blocking (Name + Address) Recall Ceiling
    print("\nTesting Recall Ceiling on Validation Split...")
    true_matches_surfaced = 0
    total_true_matches = sum(len(m) for m in val_gt.values())
    total_candidates_generated = 0
    
    baseline_predictions = {}

    t0 = time.time()
    for s1_id, gt_matches in val_gt.items():
        rec = val_s1_records.get(s1_id, {})
        n_toks = extract_name_tokens(rec.get("name", ""))
        a_toks = extract_addr_tokens(rec.get("addr", ""))

        s1_n_tids = frozenset(vocab[t] for t in n_toks if t in vocab)
        s1_a_tids = frozenset(vocab[t] for t in a_toks if t in vocab)

        cands = set()

        # Strategy 1: Name Rarest Tokens (<= 300 postings)
        if s1_n_tids:
            sorted_n = sorted(s1_n_tids, key=lambda tid: len(name_inverted_index.get(tid, ())))
            for tid in sorted_n[:2]:
                postings = name_inverted_index.get(tid)
                if postings:
                    if len(postings) <= 300:
                        cands.update(postings)
                    elif not cands:
                        cands.update(postings[:300])

        # Strategy 2: Address Rarest Tokens (e.g. house number / street name / PIN)
        if s1_a_tids:
            # Address tokens with posting length between 2 and 500 (e.g. street names, specific numbers)
            sorted_a = sorted(s1_a_tids, key=lambda tid: len(addr_inverted_index.get(tid, ())))
            for tid in sorted_a[:2]:
                postings = addr_inverted_index.get(tid)
                if postings and 2 <= len(postings) <= 500:
                    cands.update(postings)
                    break

        total_candidates_generated += len(cands)
        surfaced_eids = {target_ids[cid] for cid in cands}
        
        # Check recall ceiling
        if gt_matches:
            true_matches_surfaced += len(surfaced_eids & gt_matches)

        # Baseline Jaccard predictions (Name >= 0.6)
        matched_eids = set()
        N = len(s1_n_tids)
        if N > 0:
            for cid in cands:
                c_toks = target_name_toks[cid]
                inter = len(s1_n_tids & c_toks)
                if inter > 0:
                    union = N + len(c_toks) - inter
                    if inter / union >= 0.6:
                        matched_eids.add(target_ids[cid])
        
        baseline_predictions[s1_id] = matched_eids

    recall_ceiling = (true_matches_surfaced / total_true_matches * 100) if total_true_matches > 0 else 0
    avg_cands = total_candidates_generated / len(val_gt)

    metrics = evaluate_predictions(baseline_predictions, val_gt)

    print("=" * 65)
    print("RESULTS:")
    print(f"Total True Matches:            {total_true_matches:,}")
    print(f"True Matches Surfaced:         {true_matches_surfaced:,}")
    print(f"BLOCKING RECALL CEILING:       {recall_ceiling:.2f}%")
    print(f"Average Candidates per Entity: {avg_cands:.1f}")
    print(f"Baseline Macro F0.5:           {metrics['macro_f05']:.4f}")
    print(f"Baseline Macro Precision:      {metrics['macro_precision']:.4f}")
    print(f"Baseline Macro Recall:         {metrics['macro_recall']:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
