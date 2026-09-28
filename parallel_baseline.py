#!/usr/bin/env python3
"""
Ultra-Fast Parallel Baseline Pipeline for Amazon ML Challenge 2026.
Uses Prefix-Filtering + Fork-based Copy-On-Write Multiprocessing to process
1.73M test entities in minutes with minimal memory footprint.
"""

import os
import sys
import re
import time
import math
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "student_resource" / "dataset" / "test"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHING_OUT = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUT = OUTPUT_DIR / "candidate_pairs.tsv"

NAME_SIM_THRESHOLD = 0.6       # Jaccard threshold for final matches
CANDIDATE_SIM_THRESHOLD = 0.3  # Jaccard threshold for candidate pairs
MAX_POSTING_LEN = 3000         # Skip hyper-frequent corporate stopwords in candidate expansion
MAX_CANDIDATES_PER_ENTITY = 50 # Cap candidates per entity to keep files manageable
CHUNK_SIZE = 15000             # Batch size for parallel workers
NUM_WORKERS = min(12, os.cpu_count() or 4)

LEGAL_SUFFIXES = {
    "corp", "corporation", "pvt", "private", "ltd", "limited",
    "llc", "inc", "incorporated", "co", "company", "sarl", "sas", "llp", "sa", "gmbh",
}
STOPWORDS = {"the", "and", "of", "a", "an", "in", "to", "for", "at", "on", "by"}


# ---------------------------------------------------------------------------
# Text Normalization & Tokenization
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
# Global Shared Data (Read-only for workers via Copy-On-Write)
# ---------------------------------------------------------------------------
VOCAB = {}
TARGET_IDS = []           # Index -> entity_id (string: S2-xxx, S3-yyy)
TARGET_TOKS = []          # Index -> frozenset of token_ids
INVERTED_INDEX = {}       # token_id -> list of target_indices


def init_worker():
    """Worker initialization (shared memory inherited via fork)."""
    pass


def process_s1_batch(batch):
    """
    Process a chunk of (s1_id, business_name) tuples.
    Returns: list of (s1_id, matched_ids_str, candidate_ids_str)
    """
    results = []
    
    for s1_id, name in batch:
        toks = extract_tokens(name)
        s1_tids = frozenset(VOCAB[t] for t in toks if t in VOCAB)
        N = len(s1_tids)

        matched_ids = []
        candidate_scored = []

        if N > 0:
            # Sort tokens by posting list length (rarest first)
            sorted_tids = sorted(s1_tids, key=lambda tid: len(INVERTED_INDEX.get(tid, ())))
            
            # Prefix filtering for Jaccard >= CANDIDATE_SIM_THRESHOLD
            min_overlap = max(1, math.ceil(CANDIDATE_SIM_THRESHOLD * N))
            prefix_len = N - min_overlap + 1
            prefix_tids = sorted_tids[:prefix_len]

            # Filter tokens exceeding MAX_POSTING_LEN unless none remain
            rare_prefix = [tid for tid in prefix_tids if len(INVERTED_INDEX.get(tid, ())) <= MAX_POSTING_LEN]
            if not rare_prefix and prefix_tids:
                rare_prefix = [min(prefix_tids, key=lambda tid: len(INVERTED_INDEX.get(tid, ())))]

            # Gather candidate indices
            cands = set()
            for tid in rare_prefix:
                postings = INVERTED_INDEX.get(tid)
                if postings:
                    cands.update(postings)

            # Score candidates
            max_c_len = int(N / CANDIDATE_SIM_THRESHOLD)
            for cid in cands:
                c_toks = TARGET_TOKS[cid]
                len_c = len(c_toks)
                if len_c < min_overlap or len_c > max_c_len:
                    continue
                
                inter = len(s1_tids & c_toks)
                if inter < min_overlap:
                    continue

                union = N + len_c - inter
                jaccard = inter / union if union > 0 else 0.0

                if jaccard >= NAME_SIM_THRESHOLD:
                    matched_ids.append(TARGET_IDS[cid])
                if jaccard >= CANDIDATE_SIM_THRESHOLD:
                    candidate_scored.append((jaccard, TARGET_IDS[cid]))

        # Format candidates and ensure matches are a strict subset
        candidate_ids = [eid for _, eid in sorted(candidate_scored, reverse=True, key=lambda x: x[0])[:MAX_CANDIDATES_PER_ENTITY]]
        for mid in matched_ids:
            if mid not in candidate_ids:
                candidate_ids.append(mid)

        matched_str = ",".join(sorted(matched_ids))
        candidate_str = ",".join(sorted(candidate_ids))
        results.append((s1_id, matched_str, candidate_str))

    return results


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    print("=" * 65)
    print("Amazon ML Challenge 2026 - Ultra-Fast Parallel Baseline")
    print(f"Workers:          {NUM_WORKERS} parallel CPU processes")
    print(f"Test Directory:   {TEST_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 65)

    def get_tid(token: str) -> int:
        return VOCAB.setdefault(token, len(VOCAB))

    # 1. Index Source 2 & Source 3 into memory
    inv_dict = defaultdict(list)
    sources = [
        ("Source 2", TEST_DIR / "test_source2.tsv"),
        ("Source 3", TEST_DIR / "test_source3.tsv"),
    ]

    for label, path in sources:
        t0 = time.time()
        print(f"Loading and indexing {label} ({path.name})...")
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                parts = line.split("\t", 2)
                if not parts:
                    continue
                eid = parts[0]
                name = parts[1] if len(parts) > 1 else ""

                toks = extract_tokens(name)
                tids = frozenset(get_tid(t) for t in toks)

                idx = len(TARGET_IDS)
                TARGET_IDS.append(eid)
                TARGET_TOKS.append(tids)

                for tid in tids:
                    inv_dict[tid].append(idx)

                count += 1
                if count % 1000000 == 0:
                    print(f"  ... {count:,} records processed")
        print(f"  Finished {label}: {count:,} records in {time.time() - t0:.1f}s")

    # Freeze inverted index dict
    global INVERTED_INDEX
    INVERTED_INDEX = dict(inv_dict)
    del inv_dict  # Free temporary defaultdict overhead

    total_targets = len(TARGET_IDS)
    print(f"\nIndexed total target records: {total_targets:,}")
    print(f"Vocabulary size:              {len(VOCAB):,} unique tokens")

    # 2. Batch Source 1 records
    s1_path = TEST_DIR / "test_source1.tsv"
    print(f"\nReading Source 1 entities from {s1_path.name}...")
    t_read_s1 = time.time()

    def generate_s1_batches():
        batch = []
        with open(s1_path, "r", encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                parts = line.split("\t", 2)
                if not parts:
                    continue
                s1_id = parts[0]
                name = parts[1] if len(parts) > 1 else ""
                batch.append((s1_id, name))
                if len(batch) >= CHUNK_SIZE:
                    yield batch
                    batch = []
            if batch:
                yield batch

    # 3. Process batches in parallel with mp.Pool
    print(f"Starting parallel query execution across {NUM_WORKERS} workers...")
    t_parallel = time.time()

    total_s1 = 0
    total_with_matches = 0
    total_with_candidates = 0

    with open(MATCHING_OUT, "w", encoding="utf-8", newline="") as f_match, \
         open(CANDIDATE_OUT, "w", encoding="utf-8", newline="") as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        # Fork workers (inherits TARGET_IDS, TARGET_TOKS, INVERTED_INDEX via COW)
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=NUM_WORKERS) as pool:
            for batch_results in pool.imap(process_s1_batch, generate_s1_batches()):
                for s1_id, match_str, cand_str in batch_results:
                    f_match.write(f"{s1_id}\t{match_str}\n")
                    f_cand.write(f"{s1_id}\t{cand_str}\n")
                    total_s1 += 1
                    if match_str:
                        total_with_matches += 1
                    if cand_str:
                        total_with_candidates += 1

                if total_s1 % 200000 < CHUNK_SIZE:
                    elapsed = time.time() - t_parallel
                    qps = total_s1 / elapsed if elapsed > 0 else 0
                    print(f"  Processed {total_s1:,} / 1,732,544 entities ({qps:,.0f} queries/sec)...")

    t_total = time.time() - t_start
    print("\n" + "=" * 65)
    print("Baseline Execution Summary:")
    print(f"Total Source 1 entities processed: {total_s1:,}")
    print(f"Entities with >= 1 candidate:      {total_with_candidates:,} ({total_with_candidates / total_s1 * 100:.1f}%)")
    print(f"Entities with >= 1 match:          {total_with_matches:,} ({total_with_matches / total_s1 * 100:.1f}%)")
    print(f"Total execution time:              {t_total / 60:.2f} minutes")
    print(f"Wrote matching results:            {MATCHING_OUT}")
    print(f"Wrote candidate pairs:             {CANDIDATE_OUT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
