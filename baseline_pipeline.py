"""
Amazon ML Challenge 2026 -- Business Entity Resolution
STEP 1 BASELINE: naive token-overlap blocking + trivial rule-based matcher.

Goal: get ONE correctly-formatted submission onto the leaderboard fast.
This is deliberately simple -- swap the matching step for a real classifier
(LightGBM/XGBoost on pairwise features) once this runs cleanly end-to-end.

Run from the same folder level as your dataset/ directory:
    python3 baseline_pipeline.py
(or paste into a Colab cell -- pandas is already installed there)
"""

import re
import csv
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ---------------------------------------------------------------------------
# 0. Config -- adjust paths if your folder layout differs
# ---------------------------------------------------------------------------
TEST_DIR = Path("dataset/test")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NAME_SIM_THRESHOLD = 0.6  # Jaccard on name tokens -- raise this if you see false merges

LEGAL_SUFFIXES = {
    "corp", "corporation", "pvt", "private", "ltd", "limited",
    "llc", "inc", "incorporated", "co", "company",
}
STOPWORDS = {"the", "and", "of", "a", "an"}
ADDRESS_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "apt": "apartment",
}


# ---------------------------------------------------------------------------
# 1. Normalization
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)  # strip punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def name_tokens(name: str) -> set:
    tokens = normalize_text(name).split()
    return {t for t in tokens if t not in LEGAL_SUFFIXES and t not in STOPWORDS and len(t) > 1}


def address_tokens(addr: str) -> set:
    tokens = [ADDRESS_ABBREV.get(t, t) for t in normalize_text(addr).split()]
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


# ---------------------------------------------------------------------------
# 2. Load data (always sep="\t" -- these are TSVs, not CSVs)
# ---------------------------------------------------------------------------
def load_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    df["name_toks"] = df["business_name"].apply(name_tokens)
    df["addr_toks"] = df["business_address"].apply(address_tokens)  # not used yet -- hook for later
    return df


s1 = load_source(TEST_DIR / "test_source1.tsv")
s2 = load_source(TEST_DIR / "test_source2.tsv")
s3 = load_source(TEST_DIR / "test_source3.tsv")
print(f"S1: {len(s1)} rows | S2: {len(s2)} rows | S3: {len(s3)} rows")


# ---------------------------------------------------------------------------
# 3. Blocking -- token index over S2 + S3 name tokens (Section B in the playbook)
# ---------------------------------------------------------------------------
def build_token_index(df: pd.DataFrame) -> dict:
    index = defaultdict(set)
    for row in df.itertuples():
        for tok in row.name_toks:
            index[tok].add(row.entity_id)
    return index


idx2 = build_token_index(s2)
idx3 = build_token_index(s3)
lookup = {row.entity_id: row for row in s2.itertuples()}
lookup.update({row.entity_id: row for row in s3.itertuples()})


def candidates_for(toks: set) -> set:
    cands = set()
    for tok in toks:
        cands |= idx2.get(tok, set())
        cands |= idx3.get(tok, set())
    return cands


# ---------------------------------------------------------------------------
# 4. Trivial rule-based match: Jaccard on name tokens, thresholded.
#    Country is intentionally NOT a filter here -- it's an open set, France
#    has zero training examples, so a country-based rule would silently
#    break on exactly the entities you most need to get right.
# ---------------------------------------------------------------------------
def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


candidate_rows, match_rows = [], []

for row in s1.itertuples():
    cands = candidates_for(row.name_toks)
    candidate_rows.append((row.entity_id, ",".join(sorted(cands))))

    matches = {cid for cid in cands if jaccard(row.name_toks, lookup[cid].name_toks) >= NAME_SIM_THRESHOLD}
    match_rows.append((row.entity_id, ",".join(sorted(matches))))

print(f"Entities with >=1 candidate: {sum(1 for _, c in candidate_rows if c)}/{len(s1)}")
print(f"Entities with >=1 match:     {sum(1 for _, m in match_rows if m)}/{len(s1)}")


# ---------------------------------------------------------------------------
# 5. Write output -- exact schema the challenge expects
# ---------------------------------------------------------------------------
def write_tsv(path: Path, header: tuple, rows: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


write_tsv(OUTPUT_DIR / "matching_results.tsv", ("source1_entity_id", "matched_entity_ids"), match_rows)
write_tsv(OUTPUT_DIR / "candidate_pairs.tsv", ("source1_entity_id", "candidate_entity_ids"), candidate_rows)

print("Wrote output/matching_results.tsv and output/candidate_pairs.tsv")
print("Next: python3 utils/validate_submission.py --matching output/matching_results.tsv "
      "--candidate output/candidate_pairs.tsv --test-dir dataset/test")
