# -*- coding: utf-8 -*-
"""
feature_engineering.py
=======================
Pairwise + group-level features for the (query, candidate) pairs produced by
stage1_blocking.generate_candidates(). These feed the LightGBM classifier in
train_lightgbm.py.

Key finding from prototyping that shapes this file's design:
  Character n-gram Jaccard is FRAGILE to scrambled/transposed typos. Tested on the
  brief's own example, 'development' vs 'dveegpemnt': char-3-gram Jaccard = 0.000
  (a single reordering shifts every downstream trigram, destroying overlap), while
  Levenshtein ratio = 0.667 and token_sort_ratio = 0.667 -- both correctly recognize
  the strings as related. The reverse is also true for other noise types (word
  permutation is trivial for sorted-token comparison but can suppress a raw
  Levenshtein ratio). NO single metric dominates across noise types, which is
  exactly why this module computes several and lets the GBDT learn the weighting,
  rather than hand-picking one "best" string metric.

rapidfuzz is used when available (C++ implementation, ~50-100x faster than pure
Python at this scale) with a difflib-based fallback so the code still runs on a
minimal environment.
"""

from __future__ import annotations
import re
from typing import Optional

import numpy as np
import pandas as pd

from stage1_blocking import normalize_name, normalize_address, detect_script, strip_diacritics

try:
    from rapidfuzz import fuzz as _rf_fuzz
    from rapidfuzz.distance import Levenshtein as _rf_lev

    def token_sort_ratio(a: str, b: str) -> float:
        return _rf_fuzz.token_sort_ratio(a, b) / 100.0

    def token_set_ratio(a: str, b: str) -> float:
        return _rf_fuzz.token_set_ratio(a, b) / 100.0

    def levenshtein_ratio(a: str, b: str) -> float:
        return _rf_lev.normalized_similarity(a, b)

    _FAST_BACKEND = True
except ImportError:
    import difflib

    def token_sort_ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, " ".join(sorted(a.split())),
                                        " ".join(sorted(b.split()))).ratio()

    def token_set_ratio(a: str, b: str) -> float:
        sa, sb = set(a.split()), set(b.split())
        inter = sa & sb
        base = " ".join(sorted(inter))
        return max(
            difflib.SequenceMatcher(None, base, " ".join(sorted(sa))).ratio(),
            difflib.SequenceMatcher(None, base, " ".join(sorted(sb))).ratio(),
        )

    def levenshtein_ratio(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

    _FAST_BACKEND = False


def char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    def grams(s: str):
        s = s.replace(" ", "")
        return set(s[i:i + n] for i in range(len(s) - n + 1)) if len(s) >= n else {s}
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


DOMAIN_TLD_RE = re.compile(r"\.(com|net|org|co|in|fr|io|biz)$")


def domain_to_compact(s: str) -> Optional[str]:
    """Detect a website-domain-shaped string and reduce it to bare alnum chars, so it
    can be compared against a similarly space-stripped business name WITHOUT solving
    word segmentation ('moorebitwise' vs 'moorebitwise' from 'Moore Bitwise Inc') --
    verified: this gives Jaccard/Levenshtein = 1.0 on the brief's own example, whereas
    attempting to segment 'moorebitwise' into ['moore','bitwise'] is a harder, fragile
    sub-problem with no guaranteed win over just comparing compact character strings."""
    s = s.strip().lower()
    if not DOMAIN_TLD_RE.search(s) and "." not in s:
        return None
    s = DOMAIN_TLD_RE.sub("", s)
    return re.sub(r"[^a-z0-9]", "", s)


def name_to_compact(s: str) -> str:
    s = strip_diacritics(s.lower())
    from stage1_blocking import LEGAL_SUFFIX_RE
    s = LEGAL_SUFFIX_RE.sub("", s)
    return re.sub(r"[^a-z0-9]", "", s)


# --------------------------------------------------------------------------- #
# Pairwise feature vector for one (query, candidate) row
# --------------------------------------------------------------------------- #

PAIR_FEATURE_NAMES = [
    "blocking_score", "exact_norm_match", "exact_sorted_match",
    "token_sort_ratio", "token_set_ratio", "lev_ratio",
    "char2_jaccard", "char3_jaccard", "char4_jaccard",
    "len_char_diff", "len_token_diff", "n_shared_tokens", "token_overlap_ratio",
    "addr_text_lev_ratio", "addr_numbers_exact_match", "addr_numbers_any_overlap",
    "addr_numbers_count_diff", "country_match", "is_cross_script",
    "domain_match_score", "has_domain_signal",
]


def compute_pair_features(row: pd.Series, corpus_freq=None) -> dict:
    q_name, q_addr, q_country = row["q_business_name"], row["q_business_address"], row["q_country"]
    c_name, c_addr, c_country = row["c_business_name"], row["c_business_address"], row["c_country"]

    q_norm, q_sorted = normalize_name(q_name, corpus_freq)
    c_norm, c_sorted = normalize_name(c_name, corpus_freq)
    q_addr_n, c_addr_n = normalize_address(q_addr), normalize_address(c_addr)

    q_tokens, c_tokens = set(q_norm.split()), set(c_norm.split())
    shared = q_tokens & c_tokens
    union = q_tokens | c_tokens

    q_domain, c_domain = domain_to_compact(q_name), domain_to_compact(c_name)
    domain_score, has_domain = 0.0, 0
    if q_domain or c_domain:
        has_domain = 1
        d_side = q_domain if q_domain else name_to_compact(q_name)
        n_side = name_to_compact(c_name) if q_domain else (c_domain or name_to_compact(c_name))
        domain_score = char_ngram_jaccard(d_side, n_side, 3)

    q_nums, c_nums = set(q_addr_n["numbers"]), set(c_addr_n["numbers"])

    return {
        "exact_norm_match": int(q_norm == c_norm),
        "exact_sorted_match": int(q_sorted == c_sorted),
        "token_sort_ratio": token_sort_ratio(q_norm, c_norm),
        "token_set_ratio": token_set_ratio(q_norm, c_norm),
        "lev_ratio": levenshtein_ratio(q_norm, c_norm),
        "char2_jaccard": char_ngram_jaccard(q_norm, c_norm, 2),
        "char3_jaccard": char_ngram_jaccard(q_norm, c_norm, 3),
        "char4_jaccard": char_ngram_jaccard(q_norm, c_norm, 4),
        "len_char_diff": abs(len(q_norm) - len(c_norm)),
        "len_token_diff": abs(len(q_tokens) - len(c_tokens)),
        "n_shared_tokens": len(shared),
        "token_overlap_ratio": (len(shared) / len(union)) if union else 0.0,
        "addr_text_lev_ratio": levenshtein_ratio(q_addr_n["text"], c_addr_n["text"]),
        "addr_numbers_exact_match": int(q_nums == c_nums and len(q_nums) > 0),
        "addr_numbers_any_overlap": int(len(q_nums & c_nums) > 0),
        "addr_numbers_count_diff": abs(len(q_nums) - len(c_nums)),
        "country_match": int(q_country == c_country),
        "is_cross_script": int(detect_script(q_name) != detect_script(c_name)),
        "domain_match_score": domain_score,
        "has_domain_signal": has_domain,
    }


def build_feature_matrix(pairs_df: pd.DataFrame, corpus_freq=None, n_jobs: int = 1) -> pd.DataFrame:
    """pairs_df must have columns: query_id, target_id, blocking_score,
    q_business_name, q_business_address, q_country,
    c_business_name, c_business_address, c_country.
    Returns pairs_df joined with the pairwise feature columns AND per-query
    group-level (rank/margin/aggregate) features -- the group features matter
    disproportionately for singleton detection, since they characterize the WHOLE
    candidate set's shape, not just one pair in isolation."""
    if n_jobs == 1:
        feat_records = [compute_pair_features(r, corpus_freq) for _, r in pairs_df.iterrows()]
    else:
        from joblib import Parallel, delayed
        feat_records = Parallel(n_jobs=n_jobs)(
            delayed(compute_pair_features)(r, corpus_freq) for _, r in pairs_df.iterrows()
        )
    feat_df = pd.DataFrame(feat_records)
    out = pd.concat([pairs_df.reset_index(drop=True), feat_df], axis=1)
    out = add_group_features(out)
    return out


def add_group_features(df: pd.DataFrame, score_col: str = "blocking_score") -> pd.DataFrame:
    """Per-query aggregate features computed over its own candidate set. `rank` and
    `margin_to_next` are the two that matter most for the k=0 (singleton) decision:
    a query where the best candidate barely beats the second-best is exactly the
    ambiguous case the classifier and the F0.5-optimal thresholding step both need
    to see as a group-level signal, not just as an isolated pairwise score."""
    df = df.sort_values(["query_id", score_col], ascending=[True, False]).copy()
    g = df.groupby("query_id")[score_col]
    df["rank_in_query"] = g.rank(ascending=False, method="first").astype(int)
    df["query_max_score"] = g.transform("max")
    df["query_mean_score"] = g.transform("mean")
    df["query_std_score"] = g.transform("std").fillna(0.0)
    df["query_n_candidates"] = g.transform("count")
    next_score = g.shift(-1)
    df["margin_to_next"] = (df[score_col] - next_score).fillna(df[score_col])
    df["score_vs_query_max"] = df[score_col] / df["query_max_score"].replace(0, np.nan)
    df["score_vs_query_max"] = df["score_vs_query_max"].fillna(1.0)
    return df.reset_index(drop=True)


FULL_FEATURE_NAMES = PAIR_FEATURE_NAMES + [
    "rank_in_query", "query_max_score", "query_mean_score",
    "query_std_score", "query_n_candidates", "margin_to_next", "score_vs_query_max",
]

if __name__ == "__main__":
    demo = pd.DataFrame([
        dict(query_id="S1-1", target_id="S2-1", blocking_score=0.95,
             q_business_name="5mart Development LLC", q_business_address="0337 Main St", q_country="US",
             c_business_name="Smart Development LLC", c_business_address="337 Main St", c_country="US"),
        dict(query_id="S1-1", target_id="S3-9", blocking_score=0.40,
             q_business_name="5mart Development LLC", q_business_address="0337 Main St", q_country="US",
             c_business_name="Smart Logistics Group", c_business_address="12 Elm St", c_country="US"),
    ])
    feats = build_feature_matrix(demo)
    print(f"rapidfuzz backend in use: {_FAST_BACKEND}")
    print(feats[["query_id", "target_id", "lev_ratio", "token_sort_ratio",
                  "addr_numbers_exact_match", "rank_in_query", "margin_to_next"]])
