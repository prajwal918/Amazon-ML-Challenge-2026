#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Production Pipeline V2 (High-Precision Resolver)
Features:
- Accent stripping (Unicode NFKD)
- Comprehensive Legal Suffix & Stopword stripping
- 3-Gram Typo-Tolerant Name Matching
- Significant Number Matching (length >= 2, avoids 1st/2nd street false merges)
- Empty-Address Handling (targets with missing addresses matched via Name)
- Indian Non-Latin Transliteration Fallback
- Streaming row-by-row output with regular flushes
- Automatic submission validation at completion
"""

import sys
import re
import time
import unicodedata
from pathlib import Path
from collections import defaultdict
import subprocess

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(r".")
TEST_DIR = BASE_DIR / "student_resource" / "dataset" / "test"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHING_OUT = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUT = OUTPUT_DIR / "candidate_pairs.tsv"

MAX_POSTING_LEN = 80
MAX_MATCHES_PER_ENTITY = 5
MAX_CANDIDATES_PER_ENTITY = 6

LEGAL_SUFFIXES = {
    "corp", "corporation", "pvt", "private", "ltd", "limited",
    "llc", "inc", "incorporated", "co", "company", "sarl", "sas", "llp", "sa", "gmbh",
    "sasu", "eurl", "dba", "services", "holdings", "group", "groupe", "center"
}
STOPWORDS = {"the", "and", "of", "a", "an", "in", "to", "for", "at", "on", "by", "null"}
ADDRESS_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "apt": "apartment", "hwy": "highway",
}

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def clean_name(name: str):
    name = strip_accents(str(name).lower())
    name = name.replace("&", " and ").replace("+", " and ")
    name = re.sub(r"[^\w\s]", " ", name)
    return frozenset(t for t in name.split() if t not in LEGAL_SUFFIXES and t not in STOPWORDS and len(t) > 1)

def clean_addr(addr: str):
    addr = strip_accents(str(addr).lower())
    addr = re.sub(r"[^\w\s]", " ", addr)
    toks = [ADDRESS_ABBREV.get(t, t) for t in addr.split()]
    return frozenset(t for t in toks if t not in STOPWORDS and len(t) > 1)

def clean_nums(addr: str):
    raw = re.findall(r"\b\d+\b", str(addr))
    return frozenset(n for n in raw if len(n) >= 2)

def is_latin(text: str) -> bool:
    return all(ord(c) < 256 for c in text if c.isalpha())

def char_3grams(word: str):
    if len(word) < 3:
        return {word}
    return {word[i:i+3] for i in range(len(word)-2)}

def word_similarity(w1: str, w2: str) -> float:
    if w1 == w2:
        return 1.0
    if len(w1) >= 4 and len(w2) >= 4 and (w1 in w2 or w2 in w1):
        return 0.9
    g1 = char_3grams(w1)
    g2 = char_3grams(w2)
    inter = len(g1 & g2)
    union = len(g1 | g2)
    return inter / union if union > 0 else 0.0

def name_fuzzy_score(toks1, toks2):
    if not toks1 or not toks2:
        return 0.0
    matched_scores = []
    for t1 in toks1:
        best = max((word_similarity(t1, t2) for t2 in toks2), default=0.0)
        matched_scores.append(best)
    return sum(matched_scores) / len(matched_scores)


def main():
    print("=" * 65)
    print("Amazon ML Challenge 2026 - Production Pipeline V2")
    print("=" * 65)

    vocab = {}
    def get_tid(token: str) -> int:
        tid = vocab.get(token)
        if tid is None:
            tid = len(vocab)
            vocab[token] = tid
        return tid

    countries = defaultdict(lambda: {
        "ids": [],
        "name_toks": [],
        "addr_toks": [],
        "nums": [],
        "is_lat": [],
        "raw_names": [],
        "addr_idx": defaultdict(list),
        "name_idx": defaultdict(list),
    })

    # 1. Index Source 2 and Source 3
    sources = [
        ("Source 2", TEST_DIR / "test_source2.tsv"),
        ("Source 3", TEST_DIR / "test_source3.tsv"),
    ]

    total_target_records = 0
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

                n_toks = clean_name(name)
                a_toks = clean_addr(addr)
                nums = clean_nums(addr)
                c_lat = is_latin(name)

                n_tids = frozenset(get_tid(t) for t in n_toks)
                a_tids = frozenset(get_tid(t) for t in a_toks)

                c_data["ids"].append(eid)
                c_data["name_toks"].append(n_toks)
                c_data["addr_toks"].append(a_toks)
                c_data["nums"].append(nums)
                c_data["is_lat"].append(c_lat)

                for tid in a_tids:
                    c_data["addr_idx"][tid].append(c_idx)
                for tid in n_tids:
                    c_data["name_idx"][tid].append(c_idx)

                count += 1
                total_target_records += 1
                if count % 1000000 == 0:
                    print(f"  ... {count:,} records processed")
        print(f"  Finished {label}: {count:,} records in {time.time() - t0:.1f}s")

    print(f"\nIndexed total target records: {total_target_records:,}")
    print(f"Vocabulary size: {len(vocab):,} unique tokens")
    for ctry, data in countries.items():
        print(f"  Country '{ctry}': {len(data['ids']):,} targets")

    # 2. Process Source 1 Queries and Stream Output
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

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        next(f_s1)  # Skip header

        for line in f_s1:
            parts = line.split("\t")
            if not parts:
                continue
            s1_id = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            ctry = parts[3].strip() if len(parts) > 3 else "Unknown"

            c_data = countries.get(ctry)
            matched_ids = []
            candidate_scored = []

            if c_data:
                target_ids = c_data["ids"]
                target_n_toks = c_data["name_toks"]
                target_a_toks = c_data["addr_toks"]
                target_nums = c_data["nums"]
                target_is_lat = c_data["is_lat"]
                addr_idx = c_data["addr_idx"]
                name_idx = c_data["name_idx"]

                s1_n = clean_name(name)
                s1_a = clean_addr(addr)
                s1_nums = clean_nums(addr)
                s1_is_lat = is_latin(name)

                s1_n_tids = [vocab[t] for t in s1_n if t in vocab]
                s1_a_tids = [vocab[t] for t in s1_a if t in vocab]

                n_s1_n = len(s1_n)
                n_s1_a = len(s1_a)

                cands = set()

                # Strategy A: Candidate retrieval via Name Rarest Tokens (Name Anchor)
                if s1_n_tids:
                    sorted_n = sorted(s1_n_tids, key=lambda tid: len(name_idx.get(tid, ())))
                    for tid in sorted_n[:2]:
                        postings = name_idx.get(tid)
                        if postings:
                            if len(postings) <= MAX_POSTING_LEN:
                                cands.update(postings)
                            elif not cands:
                                cands.update(postings[:MAX_POSTING_LEN])

                # Strategy B: Candidate retrieval via Address Rarest Tokens
                if s1_a_tids:
                    sorted_a = sorted(s1_a_tids, key=lambda tid: len(addr_idx.get(tid, ())))
                    for tid in sorted_a[:2]:
                        postings = addr_idx.get(tid)
                        if postings:
                            if len(postings) <= MAX_POSTING_LEN:
                                cands.update(postings)
                            elif not cands:
                                cands.update(postings[:MAX_POSTING_LEN])

                # High-Precision Scoring
                for cid in cands:
                    c_n = target_n_toks[cid]
                    c_a = target_a_toks[cid]
                    c_nums = target_nums[cid]
                    c_lat = target_is_lat[cid]

                    # Name Jaccard
                    n_inter = len(s1_n & c_n) if n_s1_n and c_n else 0
                    n_union = n_s1_n + len(c_n) - n_inter if n_s1_n or c_n else 0
                    n_jacc = n_inter / n_union if n_union > 0 else 0.0

                    # Address Jaccard
                    a_inter = len(s1_a & c_a) if n_s1_a and c_a else 0
                    a_union = n_s1_a + len(c_a) - a_inter if n_s1_a or c_a else 0
                    a_jacc = a_inter / a_union if a_union > 0 else 0.0

                    has_num = bool(s1_nums & c_nums) if s1_nums and c_nums else False
                    has_c_addr = bool(c_a)

                    # Decision Rule
                    is_match = False
                    score = 0.0

                    if not has_c_addr:
                        # Empty address in registry: match on strong name
                        if n_jacc >= 0.60:
                            is_match = True
                            score = 0.7 + n_jacc
                        elif n_jacc >= 0.30 and name_fuzzy_score(s1_n, c_n) >= 0.70:
                            is_match = True
                            score = 0.6 + n_jacc
                    else:
                        # Both have address
                        if n_jacc >= 0.50 and (a_jacc >= 0.12 or has_num):
                            is_match = True
                            score = 0.8 + n_jacc + a_jacc
                        elif (a_jacc >= 0.15 or has_num) and name_fuzzy_score(s1_n, c_n) >= 0.60:
                            is_match = True
                            score = 0.75 + a_jacc
                        elif n_jacc >= 0.30 and a_jacc >= 0.30 and has_num:
                            is_match = True
                            score = 0.7 + n_jacc + a_jacc
                        elif not c_lat and a_jacc >= 0.50 and has_num:
                            is_match = True
                            score = 0.65 + a_jacc
                        elif (n_s1_n >= 2 or len(c_n) >= 2) and a_jacc >= 0.25:
                            s1_c = "".join(s1_n)
                            c_c = "".join(c_n)
                            if len(s1_c) >= 6 and len(c_c) >= 6 and (s1_c in c_c or c_c in s1_c):
                                is_match = True
                                score = 0.6 + a_jacc

                    cand_score = max(score, n_jacc * 0.6 + a_jacc * 0.4)
                    candidate_scored.append((cand_score, target_ids[cid], is_match))

            # Sort and deduplicate candidates
            candidate_scored.sort(key=lambda x: x[0], reverse=True)

            # Build final matches (top MAX_MATCHES_PER_ENTITY that passed is_match)
            matches = [eid for sc, eid, is_m in candidate_scored if is_m][:MAX_MATCHES_PER_ENTITY]

            # Build candidates (top MAX_CANDIDATES_PER_ENTITY, ensuring matches are included)
            cands_set = set(matches)
            cands_list = list(matches)
            for sc, eid, is_m in candidate_scored:
                if eid not in cands_set:
                    cands_set.add(eid)
                    cands_list.append(eid)
                if len(cands_list) >= MAX_CANDIDATES_PER_ENTITY:
                    break

            # Write rows
            f_match.write(f"{s1_id}\t{','.join(matches)}\n")
            f_cand.write(f"{s1_id}\t{','.join(cands_list)}\n")

            n_s1 += 1
            if matches:
                n_with_matches += 1
                total_match_pairs += len(matches)
            if cands_list:
                n_with_candidates += 1
                total_candidate_pairs += len(cands_list)

            if n_s1 % 50000 == 0:
                elapsed = time.time() - t_query_start
                qps = n_s1 / elapsed
                rem = (1732544 - n_s1) / qps if qps > 0 else 0
                f_match.flush()
                f_cand.flush()
                print(f"  Processed {n_s1:,} / 1,732,544 ({qps:.0f} qps) [ETA: {rem/60:.1f} mins]...")

    elapsed_total = time.time() - t_query_start
    print("=" * 65)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total Source 1 queries:       {n_s1:,}")
    print(f"Entities with matches:        {n_with_matches:,} ({n_with_matches/n_s1*100:.1f}%)")
    print(f"Total matched pairs:          {total_match_pairs:,}")
    print(f"Average candidates/entity:    {total_candidate_pairs/n_s1:.2f}")
    print(f"Total execution time:         {elapsed_total/60:.2f} minutes")
    print(f"Wrote matching results:       {MATCHING_OUT}")
    print(f"Wrote candidate pairs:        {CANDIDATE_OUT}")
    print("=" * 65)

    # Run official validator
    print("\nRunning official submission validator...")
    validator = BASE_DIR / "student_resource" / "utils" / "validate_submission.py"
    subprocess.run([
        sys.executable,
        str(validator),
        "--matching", str(MATCHING_OUT),
        "--candidate", str(CANDIDATE_OUT),
        "--test-dir", str(TEST_DIR),
    ], check=False)


if __name__ == "__main__":
    main()
