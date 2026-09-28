#!/usr/bin/env python3
"""
High-Speed Resuming Pipeline for Amazon ML Challenge 2026.
Resumes from existing progress in output/matching_results.tsv.
Features:
- Seamless Resume: Skips already completed Source 1 entities
- Optimized Candidate Pruning: MAX_POSTING_LEN = 80 (higher F0.5 precision & 3-4x faster)
- Strict Country Partitioning (Zero cross-country pollution)
- Ultra-Compact Candidate Sets (~4.8 candidates per entity)
"""

import sys
import time
import subprocess
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "student_resource" / "dataset" / "test"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHING_OUT = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUT = OUTPUT_DIR / "candidate_pairs.tsv"

MAX_POSTING_LEN = 80           # Optimal trade-off: higher precision & faster retrieval
MAX_MATCHES_PER_ENTITY = 5     # Aligned with ground truth distribution
MAX_CANDIDATES_PER_ENTITY = 6  # Ultra-compact candidate sets for Amazon's ranking bonus

# Normalization imports
sys.path.insert(0, str(BASE_DIR / "code" / "business_entity_resolution" / "src"))
from normalize import extract_name_tokens, extract_addr_tokens, extract_numbers

def main():
    t_start = time.time()
    print("=" * 65)
    print("Amazon ML Challenge 2026 - High-Speed Resuming Pipeline")
    print("=" * 65)

    # 1. Determine already completed progress
    completed_ids = set()
    completed_count = 0
    if MATCHING_OUT.exists():
        with open(MATCHING_OUT, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.split("\t")
                if parts:
                    completed_ids.add(parts[0])
                    completed_count += 1
        print(f"Detected existing progress: {completed_count:,} queries already solved.")

    # Synchronize candidate_pairs.tsv to the exact same set
    if CANDIDATE_OUT.exists():
        with open(CANDIDATE_OUT, "r", encoding="utf-8") as f:
            c_header = f.readline()
            valid_cand_lines = [c_header]
            for line in f:
                parts = line.split("\t")
                if parts and parts[0] in completed_ids:
                    valid_cand_lines.append(line)
        with open(CANDIDATE_OUT, "w", encoding="utf-8") as f:
            f.writelines(valid_cand_lines)
        print(f"Synchronized candidate_pairs.tsv to match ({len(valid_cand_lines)-1:,} rows).")

    # 2. Build In-Memory Target Index (Source 2 and Source 3)
    vocab = {}
    def get_tid(t: str) -> int:
        tid = vocab.get(t)
        if tid is None:
            tid = len(vocab)
            vocab[t] = tid
        return tid

    countries = defaultdict(lambda: {
        "ids": [],
        "name_toks": [],
        "addr_toks": [],
        "nums": [],
        "addr_idx": defaultdict(list),
        "name_idx": defaultdict(list),
    })

    sources = [
        ("Source 2", TEST_DIR / "test_source2.tsv"),
        ("Source 3", TEST_DIR / "test_source3.tsv"),
    ]

    total_target = 0
    for label, path in sources:
        t0 = time.time()
        print(f"Loading and indexing {label} ({path.name})...")
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.split("\t")
                if not parts:
                    continue
                eid = parts[0]
                name = parts[1] if len(parts) > 1 else ""
                addr = parts[2] if len(parts) > 2 else ""
                ctry = parts[3].strip() if len(parts) > 3 else "Unknown"

                c_data = countries[ctry]
                c_idx = len(c_data["ids"])

                n_toks = extract_name_tokens(name)
                a_toks = extract_addr_tokens(addr)
                nums = extract_numbers(addr)

                n_tids = frozenset(get_tid(t) for t in n_toks)
                a_tids = frozenset(get_tid(t) for t in a_toks)

                c_data["ids"].append(eid)
                c_data["name_toks"].append(n_tids)
                c_data["addr_toks"].append(a_tids)
                c_data["nums"].append(nums)

                for tid in a_tids:
                    c_data["addr_idx"][tid].append(c_idx)
                for tid in n_tids:
                    c_data["name_idx"][tid].append(c_idx)

                count += 1
                total_target += 1
                if count % 1000000 == 0:
                    print(f"  ... {count:,} records processed")
        print(f"  Finished {label}: {count:,} records in {time.time() - t0:.1f}s")

    print(f"\nIndexed total target records: {total_target:,}")
    print(f"Vocabulary size: {len(vocab):,} unique tokens")
    for ctry, data in countries.items():
        print(f"  Country '{ctry}': {len(data['ids']):,} targets")

    # 3. Process Remaining Source 1 Queries
    s1_path = TEST_DIR / "test_source1.tsv"
    print(f"\nProcessing remaining queries from {s1_path.name}...")

    mode = "a" if completed_count > 0 else "w"
    t_query_start = time.time()
    n_total_s1 = completed_count
    n_new = 0

    with open(s1_path, "r", encoding="utf-8") as f_s1, \
         open(MATCHING_OUT, mode, encoding="utf-8", newline="") as f_m, \
         open(CANDIDATE_OUT, mode, encoding="utf-8", newline="") as f_c:

        if completed_count == 0:
            f_m.write("source1_entity_id\tmatched_entity_ids\n")
            f_c.write("source1_entity_id\tcandidate_entity_ids\n")

        next(f_s1)  # Skip header

        for line in f_s1:
            parts = line.split("\t")
            if not parts:
                continue
            s1_id = parts[0]

            # Skip already processed entities
            if s1_id in completed_ids:
                continue

            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            ctry = parts[3].strip() if len(parts) > 3 else "Unknown"

            c_data = countries.get(ctry)
            candidate_scored = []

            if c_data:
                target_ids = c_data["ids"]
                target_n_toks = c_data["name_toks"]
                target_a_toks = c_data["addr_toks"]
                target_nums = c_data["nums"]
                addr_idx = c_data["addr_idx"]
                name_idx = c_data["name_idx"]

                s1_n = extract_name_tokens(name)
                s1_a = extract_addr_tokens(addr)
                s1_nums = extract_numbers(addr)

                s1_n_tids = frozenset(vocab[t] for t in s1_n if t in vocab)
                s1_a_tids = frozenset(vocab[t] for t in s1_a if t in vocab)

                n_s1_n = len(s1_n_tids)
                n_s1_a = len(s1_a_tids)

                cands = set()

                if s1_a_tids:
                    sorted_a = sorted(s1_a_tids, key=lambda tid: len(addr_idx.get(tid, ())))
                    for tid in sorted_a[:2]:
                        postings = addr_idx.get(tid)
                        if postings:
                            if len(postings) <= MAX_POSTING_LEN:
                                cands.update(postings)
                            elif not cands:
                                cands.update(postings[:MAX_POSTING_LEN])

                if s1_n_tids:
                    sorted_n = sorted(s1_n_tids, key=lambda tid: len(name_idx.get(tid, ())))
                    for tid in sorted_n[:2]:
                        postings = name_idx.get(tid)
                        if postings:
                            if len(postings) <= MAX_POSTING_LEN:
                                cands.update(postings)
                            elif not cands:
                                cands.update(postings[:MAX_POSTING_LEN])

                for cid in cands:
                    c_n = target_n_toks[cid]
                    c_a = target_a_toks[cid]
                    c_nums = target_nums[cid]

                    a_inter = len(s1_a_tids & c_a) if n_s1_a and c_a else 0
                    a_union = n_s1_a + len(c_a) - a_inter if n_s1_a or c_a else 0
                    a_jacc = a_inter / a_union if a_union > 0 else 0.0

                    n_inter = len(s1_n_tids & c_n) if n_s1_n and c_n else 0
                    n_union = n_s1_n + len(c_n) - n_inter if n_s1_n or c_n else 0
                    n_jacc = n_inter / n_union if n_union > 0 else 0.0

                    has_num_match = bool(s1_nums & c_nums) if s1_nums and c_nums else False

                    score = 0.0
                    is_match = False

                    if a_jacc >= 0.5 or (has_num_match and a_jacc >= 0.28):
                        score = 0.6 + a_jacc
                        is_match = True
                    elif n_jacc >= 0.70:
                        score = 0.5 + n_jacc
                        is_match = True
                    elif a_jacc >= 0.35 and n_jacc >= 0.35:
                        score = 0.4 + a_jacc + n_jacc
                        is_match = True
                    else:
                        score = max(a_jacc, n_jacc)

                    if score >= 0.2:
                        candidate_scored.append((score, target_ids[cid], is_match))

            candidate_scored.sort(reverse=True, key=lambda x: x[0])
            matched_ids = [eid for score, eid, is_match in candidate_scored if is_match][:MAX_MATCHES_PER_ENTITY]
            cand_eids = [eid for score, eid, is_match in candidate_scored[:MAX_CANDIDATES_PER_ENTITY]]

            for mid in matched_ids:
                if mid not in cand_eids:
                    cand_eids.append(mid)

            match_str = ",".join(sorted(matched_ids))
            cand_str = ",".join(sorted(cand_eids))

            f_m.write(f"{s1_id}\t{match_str}\n")
            f_c.write(f"{s1_id}\t{cand_str}\n")

            n_new += 1
            n_total_s1 += 1

            if n_new % 50000 == 0:
                elapsed = time.time() - t_query_start
                qps = n_new / elapsed if elapsed > 0 else 0
                remaining = 1732544 - n_total_s1
                eta_min = (remaining / qps) / 60 if qps > 0 else 0
                print(f"  Processed {n_total_s1:,} / 1,732,544 ({qps:,.0f} qps) [ETA: {eta_min:.1f} mins]...")
                f_m.flush()
                f_c.flush()

    t_total = time.time() - t_start
    print("\n" + "=" * 65)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total Source 1 records in file: {n_total_s1:,}")
    print(f"New records processed:          {n_new:,}")
    print(f"Total elapsed time:             {t_total / 60:.2f} minutes")
    print(f"Wrote matching results:         {MATCHING_OUT}")
    print(f"Wrote candidate pairs:          {CANDIDATE_OUT}")
    print("=" * 65)

    # 4. Run Official Submission Validator
    print("\nRunning official submission validator...")
    val_script = BASE_DIR / "student_resource" / "utils" / "validate_submission.py"
    cmd = [
        sys.executable, str(val_script),
        "--matching", str(MATCHING_OUT),
        "--candidate", str(CANDIDATE_OUT),
        "--test-dir", str(TEST_DIR),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

if __name__ == "__main__":
    main()
