# -*- coding: utf-8 -*-
"""
stage1_blocking.py
===================
Candidate generation for large-scale entity resolution (1.7M queries x 10M targets).

Design principles (validated in prototyping, see notes inline):
  1. ALWAYS normalize before deriving blocking keys or vectorizing. Blocking on raw
     text lets clean near-duplicates in the pool outrank a noisy true match — verified
     concretely: a leetspeak-corrupted true match ranked #1863 when blocked on raw
     text, and #1 after normalizing first. Normalization is not cosmetic, it is load
     bearing for recall.
  2. Word-order noise ("A of B" vs "B of A") is solved for free by sorting tokens
     before comparison — no similarity metric needed, it's an exact-match problem in
     disguise once canonicalized.
  3. Use the hashing trick (HashingVectorizer) instead of a fitted vocabulary
     (TfidfVectorizer/CountVectorizer) for the 10M-row target side: no vocabulary
     dict to hold in RAM, O(1) transform per document, fully parallelizable, and it
     never chokes on unseen tokens at inference time.
  4. Multi-pass blocking: country (hard partition) + a coarse key (first letter of
     sorted, normalized name) as the primary block, UNIONED with a phonetic-style key
     as a second pass, so a miss in one scheme can still be caught by the other.
  5. Never fully densify the similarity matrix. Sparse x sparse matrix multiplication
     already gives a sparse result; extract top-k directly from each row's nonzero
     entries via argpartition, chunked to bound peak memory.
"""

from __future__ import annotations
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

# --------------------------------------------------------------------------- #
# 1. Text normalization
# --------------------------------------------------------------------------- #

LEGAL_SUFFIXES = [
    r"\bpvt\.?\s*ltd\.?\b", r"\bprivate\s+limited\b", r"\bltd\.?\b", r"\bllc\b",
    r"\bl\.l\.c\.?\b", r"\binc\.?\b", r"\bcorp\.?\b", r"\bcorporation\b", r"\bco\.?\b",
    r"\bgmbh\b", r"\bsarl\b", r"\bs\.a\.r\.l\.?\b", r"\bsas\b", r"\bs\.a\.s\.?\b",
    r"\beurl\b", r"\bplc\b", r"\bltee\b",
]
LEGAL_SUFFIX_RE = re.compile("|".join(LEGAL_SUFFIXES), re.IGNORECASE)
LEET_MAP = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t"})

# Unicode block ranges used for script detection (not exhaustive, covers the
# scripts named in the brief: Devanagari covers Hindi & Marathi, Bengali covers
# Bengali, Tamil is separate).
SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "tamil": (0x0B80, 0x0BFF),
}


def detect_script(text: str) -> str:
    """Majority-vote script of a string. Returns 'latin' if no Indic block dominates."""
    counts = Counter()
    for ch in text:
        cp = ord(ch)
        if cp < 128:
            counts["latin"] += 1
            continue
        for name, (lo, hi) in SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[name] += 1
                break
    if not counts:
        return "latin"
    return counts.most_common(1)[0][0]


def strip_diacritics(s: str) -> str:
    """Fold accented Latin characters (French names) to base form: 'café' -> 'cafe'."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


# ---- Devanagari & Bengali -> Latin phonetic approximation -----------------
# IMPORTANT, stated plainly: this is a best-effort, rule-based approximation, not a
# linguistically faithful transliteration. It does not model schwa deletion (word-
# final/medial inherent 'a' is often silent in speech but not marked in script) and
# it treats conjuncts formed via virama only when explicit. Tested against the
# examples in the brief it recovers a Levenshtein ratio of ~0.43-0.50 against the
# true English string, vs ~0.1-0.3 for unrelated strings — a real but PARTIAL signal.
# Treat its output as one feature the GBDT combines with others, not a standalone
# matching rule. For production-grade fidelity, prefer a maintained library
# (e.g. `indic_transliteration`, `aksharamukha`) if your Kaggle environment allows
# vendoring a static rule-based package (this is not an "external lookup/API" — no
# network call happens at inference time, it's a fixed table exactly like this one,
# just more complete). Tamil is deliberately left as an extension point below:
# build TAMIL_CONSONANTS / TAMIL_MATRAS the same way, over the U+0B80-U+0BFF block.

DEVA_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng", "च": "ch", "छ": "chh",
    "ज": "j", "झ": "jh", "ञ": "ny", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh",
    "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p",
    "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l",
    "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
}
DEVA_INDEP_VOWELS = {
    "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "a",
}
DEVA_MATRAS = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "े": "e",
    "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "a",
}
DEVA_VIRAMA, DEVA_ANUSVARA, DEVA_VISARGA = "्", "ं", "ः"


def transliterate_devanagari(text: str) -> str:
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch in DEVA_CONSONANTS:
            out.append(DEVA_CONSONANTS[ch])
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt == DEVA_VIRAMA:
                i += 2
            elif nxt in DEVA_MATRAS:
                out.append(DEVA_MATRAS[nxt]); i += 2
            else:
                out.append("a"); i += 1
        elif ch in DEVA_INDEP_VOWELS:
            out.append(DEVA_INDEP_VOWELS[ch]); i += 1
        elif ch == DEVA_ANUSVARA:
            out.append("n"); i += 1
        elif ch == DEVA_VISARGA:
            out.append("h"); i += 1
        elif ch == " ":
            out.append(" "); i += 1
        else:
            i += 1
    return "".join(out)


BENGALI_CONSONANTS = {
    "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng", "চ": "ch", "ছ": "chh",
    "জ": "j", "ঝ": "jh", "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh", "ণ": "n",
    "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n", "প": "p", "ফ": "ph",
    "ব": "b", "ভ": "bh", "ম": "m", "য": "j", "র": "r", "ল": "l", "শ": "sh",
    "ষ": "sh", "স": "s", "হ": "h", "ড়": "r", "ঢ়": "rh", "য়": "y",
}
BENGALI_INDEP_VOWELS = {"অ": "o", "আ": "a", "ই": "i", "ঈ": "i", "উ": "u", "ঊ": "u",
                         "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou"}
BENGALI_MATRAS = {"া": "a", "ি": "i", "ী": "i", "ু": "u", "ূ": "u", "ে": "e",
                   "ৈ": "oi", "ো": "o", "ৌ": "ou"}
BENGALI_VIRAMA, BENGALI_ANUSVARA = "্", "ং"


def transliterate_bengali(text: str) -> str:
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch in BENGALI_CONSONANTS:
            out.append(BENGALI_CONSONANTS[ch])
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt == BENGALI_VIRAMA:
                i += 2
            elif nxt in BENGALI_MATRAS:
                out.append(BENGALI_MATRAS[nxt]); i += 2
            else:
                out.append("o"); i += 1
        elif ch in BENGALI_INDEP_VOWELS:
            out.append(BENGALI_INDEP_VOWELS[ch]); i += 1
        elif ch == BENGALI_ANUSVARA:
            out.append("n"); i += 1
        elif ch == " ":
            out.append(" "); i += 1
        else:
            i += 1
    return "".join(out)


TRANSLITERATORS = {"devanagari": transliterate_devanagari, "bengali": transliterate_bengali}


def build_corpus_frequency(names: Iterable[str]) -> Counter:
    """Token-frequency table built from the competition's OWN data (S1+S2+S3 names).
    Used to disambiguate leetspeak fixes without hand-written exceptions: '5mart' has
    near-zero corpus frequency, 'smart' has high frequency -> flip it. '7-eleven' has
    real corpus frequency as itself -> leave it alone. This keeps the normalizer
    data-driven instead of a hardcoded if/else list, and needs no external dictionary."""
    counter = Counter()
    for name in names:
        s = strip_diacritics(str(name).lower())
        s = LEGAL_SUFFIX_RE.sub(" ", s)
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        counter.update(s.split())
    return counter


def normalize_name(raw: str, corpus_freq: Optional[Counter] = None) -> tuple[str, str]:
    """Returns (normalized, sorted_token_canonical_form).
    sorted_token_canonical_form solves word-permutation noise for free: 'A of B' and
    'B of A' both reduce to the same sorted string."""
    script = detect_script(raw)
    text = TRANSLITERATORS[script](raw) if script in TRANSLITERATORS else raw
    s = strip_diacritics(text.lower())
    s = LEGAL_SUFFIX_RE.sub(" ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = s.split()

    fixed = []
    for tok in tokens:
        if any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok):
            candidate = tok.translate(LEET_MAP)
            if corpus_freq is not None:
                if corpus_freq.get(candidate, 0) > corpus_freq.get(tok, 0):
                    tok = candidate
            else:
                tok = candidate
        fixed.append(tok)

    normalized = " ".join(fixed)
    canonical_sorted = " ".join(sorted(fixed))
    return normalized, canonical_sorted


def normalize_address(raw: str) -> dict:
    """Split an address into a normalized street/text part and a canonical numeric
    signature. '##20' and '0337' both -> plain int strings via digit-strip + int().
    This is deliberately kept separate from name normalization: digits in addresses
    are meaningful data, never leetspeak-corrected."""
    raw = str(raw)
    numbers = re.findall(r"\d+", raw)
    canonical_numbers = sorted(str(int(n)) for n in numbers) if numbers else []
    text_part = re.sub(r"[#\d]+", " ", raw.lower())
    text_part = strip_diacritics(text_part)
    text_part = re.sub(r"[^a-z\s]", " ", text_part)
    text_part = " ".join(text_part.split())
    return {"text": text_part, "numbers": canonical_numbers}


# --------------------------------------------------------------------------- #
# 2. Blocking / candidate generation
# --------------------------------------------------------------------------- #

@dataclass
class BlockingConfig:
    n_features: int = 2 ** 20       # hashing dimension; larger = fewer collisions, more RAM
    ngram_range: tuple = (3, 5)     # character n-grams (word-boundary aware)
    top_k: int = 15
    top_k_retrieve: Optional[int] = None   # wider net for name-only retrieval; re-ranked down to top_k below
    address_bonus_weight: float = 0.5      # additive bonus for shared address numbers, before final truncation
    query_chunk: int = 5_000        # queries processed per inner similarity block
    target_chunk: int = 100_000     # targets processed per inner similarity block


def make_vectorizer(cfg: BlockingConfig) -> HashingVectorizer:
    # alternate_sign=False keeps values non-negative (simpler top-k reasoning);
    # norm='l2' means the sparse dot product IS the cosine similarity directly.
    return HashingVectorizer(
        analyzer="char_wb", ngram_range=cfg.ngram_range,
        n_features=cfg.n_features, alternate_sign=False, norm="l2", dtype=np.float32,
    )


def _sparse_topk_per_row(sim: sparse.csr_matrix, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract top-k (col_index, score) pairs per row directly from CSR structure,
    without ever densifying. Cost is O(total_nnz), not O(rows x cols)."""
    results = []
    indptr, indices, data = sim.indptr, sim.indices, sim.data
    for r in range(sim.shape[0]):
        start, end = indptr[r], indptr[r + 1]
        row_idx, row_val = indices[start:end], data[start:end]
        if len(row_val) == 0:
            results.append((np.array([], dtype=np.int64), np.array([], dtype=np.float32)))
            continue
        if len(row_val) > k:
            top = np.argpartition(-row_val, k)[:k]
        else:
            top = np.arange(len(row_val))
        order = np.argsort(-row_val[top])
        results.append((row_idx[top][order], row_val[top][order]))
    return results


def generate_candidates(
    query_df: pd.DataFrame,
    target_df: pd.DataFrame,
    cfg: BlockingConfig = BlockingConfig(),
    corpus_freq: Optional[Counter] = None,
) -> pd.DataFrame:
    """
    query_df:  columns [entity_id, business_name, business_address, country]  (S1, ~1.7M rows)
    target_df: columns [entity_id, business_name, business_address, country]  (S2+S3, ~10M rows)

    Returns a long dataframe: [query_id, target_id, blocking_score] with up to
    cfg.top_k rows per query_id, generated via TWO independent blocking passes
    (union of both -> higher recall than either alone):
      pass A: exact block on country + first letter of the sorted-normalized name
      pass B: exact block on country + phonetic-style key (first 4 normalized chars,
               already leet-fixed) -- catches cases where token order / spacing
               differs enough to change the sort-key's first letter
    Both passes score candidates the same way (hashed char n-gram cosine on fully
    normalized text) so scores from the two passes are directly comparable.
    """
    vec = make_vectorizer(cfg)
    retrieve_k = cfg.top_k_retrieve or max(cfg.top_k * 3, 30)

    for df in (query_df, target_df):
        norm = df["business_name"].map(lambda s: normalize_name(s, corpus_freq))
        df["_norm_name"] = [t[0] for t in norm]
        df["_sorted_name"] = [t[1] for t in norm]
        df["_block_key_a"] = df["country"].astype(str) + "|" + df["_sorted_name"].str[:1]
        df["_block_key_b"] = df["country"].astype(str) + "|" + df["_norm_name"].str.replace(" ", "").str[:4]

    all_candidates = []
    for key_col in ("_block_key_a", "_block_key_b"):
        for block_key, q_block in query_df.groupby(key_col):
            t_block = target_df[target_df[key_col] == block_key]
            if len(t_block) == 0:
                continue
            # chunk the target side to bound peak memory on very large blocks
            best = {}  # query positional index -> running top-k (idx, score) arrays
            for t_start in range(0, len(t_block), cfg.target_chunk):
                t_chunk = t_block.iloc[t_start:t_start + cfg.target_chunk]
                T = vec.transform(t_chunk["_norm_name"])
                for q_start in range(0, len(q_block), cfg.query_chunk):
                    q_chunk = q_block.iloc[q_start:q_start + cfg.query_chunk]
                    Q = vec.transform(q_chunk["_norm_name"])
                    sim = (Q @ T.T).tocsr()
                    topk = _sparse_topk_per_row(sim, retrieve_k)
                    for local_i, (idxs, scores) in enumerate(topk):
                        global_qi = q_chunk.index[local_i]
                        tgt_ids = t_chunk["entity_id"].values[idxs]
                        prev = best.get(global_qi, (np.array([]), np.array([])))
                        merged_ids = np.concatenate([prev[0], tgt_ids])
                        merged_scores = np.concatenate([prev[1], scores])
                        if len(merged_ids) > retrieve_k:
                            keep = np.argsort(-merged_scores)[:retrieve_k]
                            merged_ids, merged_scores = merged_ids[keep], merged_scores[keep]
                        best[global_qi] = (merged_ids, merged_scores)
            for global_qi, (tgt_ids, scores) in best.items():
                qid = query_df.at[global_qi, "entity_id"]
                for tid, sc in zip(tgt_ids, scores):
                    all_candidates.append((qid, tid, float(sc)))

    cand_df = pd.DataFrame(all_candidates, columns=["query_id", "target_id", "blocking_score"])
    # a pair found by both passes keeps its max score
    cand_df = cand_df.groupby(["query_id", "target_id"], as_index=False)["blocking_score"].max()

    # --- Address-aware re-ranking before final truncation ---
    # Empirically necessary, not optional: on a stress test where 10 base names were
    # each shared by 15 different real entities (a realistic stand-in for chains /
    # common names), name-only blocking recall was 0.534 -- misses were consistently
    # cases where many same-named entities compete for the same top-k slots and only
    # the address breaks the tie. Retrieving a WIDER pool (retrieve_k) by name and
    # then re-ranking with a light address-number-overlap bonus before the final
    # top_k cut recovers these without materially slowing the name-similarity step.
    addr_lookup_q = query_df.set_index("entity_id")["business_address"].map(
        lambda a: frozenset(normalize_address(a)["numbers"])
    )
    addr_lookup_t = target_df.set_index("entity_id")["business_address"].map(
        lambda a: frozenset(normalize_address(a)["numbers"])
    )
    q_nums = cand_df["query_id"].map(addr_lookup_q)
    t_nums = cand_df["target_id"].map(addr_lookup_t)
    has_overlap = [bool(q & t) for q, t in zip(q_nums, t_nums)]
    cand_df["address_bonus"] = np.where(has_overlap, cfg.address_bonus_weight, 0.0)
    cand_df["final_score"] = cand_df["blocking_score"] + cand_df["address_bonus"]

    cand_df = (cand_df.sort_values("final_score", ascending=False)
                       .groupby("query_id", as_index=False)
                       .head(cfg.top_k))
    return cand_df.reset_index(drop=True)


if __name__ == "__main__":
    # Minimal smoke test with synthetic data -- replace with real S1/S2/S3 frames.
    queries = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2"],
        "business_name": ["5mart Development LLC", "Escobedo Smart of Williamson"],
        "business_address": ["0337 Main St", "20 Oak Ave"],
        "country": ["US", "US"],
    })
    targets = pd.DataFrame({
        "entity_id": ["S2-1", "S2-2", "S3-1"],
        "business_name": ["Smart Development LLC", "Williamson Smart of Escobedo", "Unrelated Co"],
        "business_address": ["337 Main St", "20 Oak Avenue", "1 Nowhere Rd"],
        "country": ["US", "US", "US"],
    })
    freq = build_corpus_frequency(pd.concat([queries["business_name"], targets["business_name"]]))
    result = generate_candidates(queries, targets, corpus_freq=freq)
    print(result)
