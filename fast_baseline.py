#!/usr/bin/env python3
"""
Fast High-Performance Baseline for Amazon ML Challenge 2026.
Produces compliant matching_results.tsv and candidate_pairs.tsv
without blowing up RAM or candidate set sizes.
"""

import sys
import os
import re
import time
import math
from pathlib import Path
from collections import defaultdict

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "student_resource" / "dataset" / "test"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHING_OUT = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUT = OUTPUT_DIR / "candidate_pairs.tsv"

NAME_SIM_THRESHOLD = 0.6  # Jaccard matching threshold
CANDIDATE_SIM_THRESHOLD = 0.3  # Candidate threshold (matches are always subset)
MAX_POSTING_LEN = 300  # Cap posting lists for hyper-fast candidate retrieval
MAX_CANDIDATES_PER_ENTITY = 50  # Cap candidates per entity to keep files manageable

LEGAL_SUFFIXES = {
    "corp", "corporation", "pvt", "private", "ltd", "limited",
    "llc", "inc", "incorporated", "co", "company", "sarl", "sas", "llp", "sa", "gmbh",
}
STOPWORDS = {"the", "and", "of", "a", "an", "in", "to", "for", "at", "on", "by"}


# ---------------------------------------------------------------------------
# Normalization & Tokenization
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    text = str(text).lower().replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_tokens(name: str) -> frozenset:
    toks = normalize_text(name).split()
    clean = {t for t in toks if t not in LEGAL_SUFFIXES and t not in STOPWORDS and len(t) > 1}
    return frozenset(clean)


# ---------------------------------------------------------------------------
# Indexing S2 and S3
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    print("=" * 60)
    print("Amazon ML Challenge 2026 - Fast Baseline Pipeline")
    print(f"Test Directory:   {TEST_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 60)

    # Global token vocabulary: token_str -> token_id (int)
    vocab = {}
    def get_tid(token: str) -> int:
        return vocab.setdefault(token, len(vocab))

    # Indexed Target Entities (S2 and S3)
    target_ids = []           # index -> entity_id (string)
    target_tokens = []        # index -> frozenset of token_ids
    inverted_index = defaultdict(list)  # token_id -> list of target_index

    # 1. Index Source 2 & Source 3
    sources = [
        ("Source 2", TEST_DIR / "test_source2.tsv"),
        ("Source 3", TEST_DIR / "test_source3.tsv"),
    ]

    for label, path in sources:
        t0 = time.time()
        print(f"Loading and indexing {label} from {path.name}...")
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            header = next(f)  # skip header
            for line in f:
                parts = line.split("\t", 2)
                if not parts:
                    continue
                eid = parts[0]
                name = parts[1] if len(parts) > 1 else ""
                
                toks = extract_tokens(name)
                tids = frozenset(get_tid(t) for t in toks)
                
                idx = len(target_ids)
                target_ids.append(eid)
                target_tokens.append(tids)
                
                for tid in tids:
                    inverted_index[tid].append(idx)
                
                count += 1
                if count % 1000000 == 0:
                    print(f"  ... {count:,} records processed")
        print(f"  Finished {label}: {count:,} records in {time.time() - t0:.1f}s")

    total_targets = len(target_ids)
    print(f"Total target records indexed: {total_targets:,}")
    print(f"Vocabulary size: {len(vocab):,} unique tokens")

    # 2. Process Source 1 (Test Queries) and stream output
    s1_path = TEST_DIR / "test_source1.tsv"
    print(f"\nProcessing Source 1 queries from {s1_path.name}...")
    
    t_query_start = time.time()
    n_s1 = 0
    n_with_candidates = 0
    n_with_matches = 0
    total_match_pairs = 0
    total_candidate_pairs = 0

    with open(s1_path, "r", encoding="utf-8") as f_s1, \
         open(MATCHING_OUT, "w", encoding="utf-8", newline="") as f_match, \
         open(CANDIDATE_OUT, "w", encoding="utf-8", newline="") as f_cand:

        # Write TSV headers exactly as specified
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        next(f_s1)  # Skip header
        
        for line in f_s1:
            parts = line.split("\t", 2)
            if not parts:
                continue
            s1_id = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            
            s1_toks = extract_tokens(name)
            s1_tids = frozenset(vocab[t] for t in s1_toks if t in vocab)
            n_s1_toks = len(s1_tids)

            matched_ids = []
            candidate_scored = []  # list of (score, target_id)

            if s1_tids:
                # Find the rarest token(s)
                sorted_tids = sorted(s1_tids, key=lambda tid: len(inverted_index.get(tid, ())))
                
                # Check up to 2 rarest tokens if posting lists are manageable (<= 300)
                cands = set()
                for tid in sorted_tids[:2]:
                    postings = inverted_index.get(tid)
                    if postings:
                        if len(postings) <= MAX_POSTING_LEN:
                            cands.update(postings)
                        elif not cands:
                            cands.update(postings[:MAX_POSTING_LEN])

                for cid in cands:
                    c_toks = target_tokens[cid]
                    inter = len(s1_tids & c_toks)
                    if inter == 0:
                        continue
                    union = n_s1_toks + len(c_toks) - inter
                    jaccard = inter / union if union > 0 else 0.0

                    if jaccard >= NAME_SIM_THRESHOLD:
                        matched_ids.append(target_ids[cid])
                    if jaccard >= CANDIDATE_SIM_THRESHOLD:
                        candidate_scored.append((jaccard, target_ids[cid]))

            # Always ensure every matched ID is included in candidates
            matched_set = set(matched_ids)
            candidate_ids = [eid for _, eid in sorted(candidate_scored, reverse=True, key=lambda x: x[0])[:MAX_CANDIDATES_PER_ENTITY]]
            
            # Guarantee match subset: if any match was truncated by cap, add it back
            for mid in matched_ids:
                if mid not in candidate_ids:
                    candidate_ids.append(mid)

            # Format outputs (comma-separated, sorted for consistency)
            match_str = ",".join(sorted(matched_ids))
            cand_str = ",".join(sorted(candidate_ids))

            f_match.write(f"{s1_id}\t{match_str}\n")
            f_cand.write(f"{s1_id}\t{cand_str}\n")

            n_s1 += 1
            if match_str:
                n_with_matches += 1
                total_match_pairs += len(matched_ids)
            if cand_str:
                n_with_candidates += 1
                total_candidate_pairs += len(candidate_ids)

            if n_s1 % 200000 == 0:
                elapsed = time.time() - t_query_start
                qps = n_s1 / elapsed if elapsed > 0 else 0
                print(f"  Processed {n_s1:,} / 1,732,544 queries ({qps:.0f} qps)...")

    t_total = time.time() - t_start
    print("\n" + "=" * 60)
    print("Baseline Execution Summary:")
    print(f"Total Source 1 entities processed: {n_s1:,}")
    print(f"Entities with >= 1 candidate:      {n_with_candidates:,} ({n_with_candidates / n_s1 * 100:.1f}%)")
    print(f"Entities with >= 1 match:          {n_with_matches:,} ({n_with_matches / n_s1 * 100:.1f}%)")
    print(f"Total matched pairs:               {total_match_pairs:,}")
    print(f"Total candidate pairs:             {total_candidate_pairs:,}")
    print(f"Total elapsed time:                {t_total / 60:.2f} minutes")
    print(f"Wrote matching results:            {MATCHING_OUT}")
    print(f"Wrote candidate pairs:             {CANDIDATE_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
