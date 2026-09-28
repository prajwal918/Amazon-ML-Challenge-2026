#!/usr/bin/env python3
"""
================================================================================
Amazon ML Challenge 2026 -- Multi-Source Business Entity Resolution Pipeline
================================================================================

For each Source-1 query, finds every matching record in the Source-2/Source-3
pool, optimizing macro-averaged F0.5 across all queries.

ARCHITECTURE
---------------------------------------------------------------------
  1. Normalization   -- untangles the documented corruption types (legal-
                         suffix swaps, DBA aliases, handles, domains, metadata
                         injection, leetspeak, Indic cross-script
                         transliteration) into a small set of comparable
                         string variants per record.
  2. Blocking        -- an inverted index (name tokens, a prefix fallback,
                         and house-number+state address keys) retrieves a
                         bounded candidate set per query in better-than-
                         quadratic time. Common tokens are down-weighted
                         (require corroboration) or dropped, never blindly
                         capped -- V11's capped-postings bug is exactly the
                         failure mode this avoids.
  3. Feature scoring -- ~12 name/address similarity features per
                         (query, candidate) pair, computed with rapidfuzz.
  4. Classification   -- a LightGBM model learns the precision/recall
                         trade-off from labeled pairs instead of hand-tuned
                         thresholds (V11/V14's `nsim >= 70` style rules).
  5. Decision rule    -- per query, choosing which candidates to predict is
                         the classic "maximize expected F-beta" problem; it
                         has a closed form in the number of true positives
                         and total true count, which this module maximizes
                         using the model's own probabilities as the plug-in
                         estimate. This adapts per query (0, 1, or several
                         matches) instead of applying one global threshold.

USAGE
---------------------------------------------------------------------
  Train:
    python er_pipeline.py train \
        --source1 train_source1.tsv --targets train_targets.tsv \
        --ground-truth train_ground_truth.tsv --model-out model.pkl

  Predict (writes matching_results.tsv and candidate_pairs.tsv):
    python er_pipeline.py predict \
        --source1 test_source1.tsv --targets source23_pool.tsv \
        --model model.pkl --output-dir ./output

Both subcommands accept --n-jobs (default: all cores) and --max-candidates
(default: 40, the per-query cap after blocking+pre-rank).

OUTPUT FILE SEMANTICS (assumption -- the prompt didn't fully specify this)
---------------------------------------------------------------------
  matching_results.tsv -- the final, precision-tuned prediction: exactly the
                           entity_ids select_matches() decided to keep.
  candidate_pairs.tsv   -- the broader post-blocking, post-cap candidate set
                           for the same query, i.e. everything that reached
                           the classifier, before the decision rule narrowed
                           it down. If the competition instead wants only the
                           pairs the classifier scored above some fixed
                           probability, filter pairs_df on that threshold
                           before writing -- the model's probabilities are
                           already attached to every scored pair.

Multiprocessing uses fork-based worker processes (Linux default, including
Kaggle) so the target index is shared copy-on-write rather than re-pickled
per worker. On a spawn-default platform (Windows, some macOS setups), pass
--n-jobs 1, or run inside WSL/Docker.
================================================================================
"""
import argparse
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode as _unidecode

try:
    import joblib
except ImportError:
    joblib = None

# ==============================================================================
# SECTION 1 -- NAME NORMALIZATION
# ==============================================================================

LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "pvt", "llc", "l.l.c",
    "incorporated", "inc", "corp", "corporation", "services", "center",
    "centre", "holdings", "group", "enterprises", "solutions", "llp",
    "l.l.p", "sarl", "sas", "ltd", "limited", "co", "company", "pllc", "plc",
]
_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(s) for s in LEGAL_SUFFIXES), key=len, reverse=True)) + r")\.?\b",
    re.IGNORECASE,
)
_METADATA_RE = re.compile(r"\(\s*id\s*[:#]?\s*\d+\s*\)", re.IGNORECASE)
_HANDLE_RE = re.compile(r"(?<![a-z0-9])@")
_DOMAIN_RE = re.compile(r"\.(com|net|org|io|co|in|biz)\b", re.IGNORECASE)
_DBA_RE = re.compile(r"\bd\.?\s*b\.?\s*a\.?\s*:?\s*", re.IGNORECASE)
_LEET_MAP = str.maketrans({"5": "s", "3": "e", "1": "i", "0": "o", "4": "a", "7": "t", "8": "b"})
_NONALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")

_SCRIPT_RANGES = []
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _translit
    _SCRIPT_RANGES = [
        (0x0900, 0x097F, sanscript.DEVANAGARI),
        (0x0B80, 0x0BFF, sanscript.TAMIL),
        (0x0980, 0x09FF, sanscript.BENGALI),
        (0x0C00, 0x0C7F, sanscript.TELUGU),
        (0x0C80, 0x0CFF, sanscript.KANNADA),
    ]
    _HAVE_INDIC = True
except Exception:
    _HAVE_INDIC = False


def _token_script(token):
    for ch in token:
        code = ord(ch)
        for lo, hi, scheme in _SCRIPT_RANGES:
            if lo <= code <= hi:
                return scheme
    return None


def romanize_mixed(text):
    """Romanize only tokens actually in an Indic script; leave Latin/ASCII
    tokens untouched (mixed-script strings like 'Raj Investments <tamil
    transliteration of LLP>' are common in the data)."""
    if not text or not _HAVE_INDIC:
        return text
    out = []
    for tok in text.split():
        scheme = _token_script(tok)
        if scheme:
            try:
                out.append(_translit(tok, scheme, sanscript.ITRANS))
            except Exception:
                out.append(tok)
        else:
            out.append(tok)
    return " ".join(out)


def _clean_basic(s):
    return _WS_RE.sub(" ", _NONALNUM_RE.sub(" ", s)).strip()


def normalize_name(raw):
    """Returns comparison-ready variants of a business name:
      full    - cleaned, lowercased, accents stripped, romanized if needed
      core    - full with legal suffixes removed
      leet    - core with common leetspeak substitutions undone (only
                generated when digits are actually present, so genuinely
                digit-bearing brand names are never corrupted)
      dba_alt - the alternate name after a DBA marker, if any
    """
    if not isinstance(raw, str) or not raw.strip():
        return {"full": "", "core": "", "leet": "", "dba_alt": ""}

    s = raw.strip()
    dba_alt = ""
    m = _DBA_RE.search(s)
    if m:
        after = s[m.end():].strip()
        if after:
            dba_alt = _clean_basic(_unidecode(after).lower())
            dba_alt = _WS_RE.sub(" ", _SUFFIX_RE.sub(" ", dba_alt)).strip()
        s = s[: m.start()].strip()

    s = _METADATA_RE.sub(" ", s)
    s = _HANDLE_RE.sub(" ", s)
    s = _DOMAIN_RE.sub(" ", s)
    s = romanize_mixed(s)
    s = _unidecode(s).lower()
    full = _clean_basic(s)
    core = _WS_RE.sub(" ", _SUFFIX_RE.sub(" ", full)).strip()
    leet = ""
    if any(ch.isdigit() for ch in core):
        candidate = core.translate(_LEET_MAP)
        if candidate != core:
            leet = candidate

    return {"full": full, "core": core, "leet": leet, "dba_alt": dba_alt}


# ==============================================================================
# SECTION 2 -- ADDRESS NORMALIZATION
# ==============================================================================

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}
IN_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar",
    "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana", "hp": "himachal pradesh",
    "jh": "jharkhand", "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh",
    "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland",
    "or": "odisha", "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu",
    "tg": "telangana", "tr": "tripura", "up": "uttar pradesh", "uk": "uttarakhand",
    "wb": "west bengal", "dl": "delhi",
}
# NB: a couple of 2-letter codes collide across the two country schemes (e.g.
# "or" = Oregon in the US, Odisha in India; "ga" = Georgia/Goa). Since
# blocking and scoring both restrict candidates to the same `country` field
# first, this never causes cross-country confusion in practice.
STATE_ABBR = {**US_STATES, **IN_STATES}
FULL_TO_ABBR = {v: k for k, v in STATE_ABBR.items()}
# pre-renaming / colloquial names for present-day states: any full-name
# string recognized in address text resolves to this canonical form, so
# "Orissa" (pre-2011) and "Odisha" compare equal.
FULL_NAME_CANON = {full: full for full in FULL_TO_ABBR}
FULL_NAME_CANON.update({"orissa": "odisha", "uttaranchal": "uttarakhand", "pondicherry": "puducherry"})

_NUM_RE = re.compile(r"\d+")


def normalize_address(raw):
    """Returns house_number (leading zeros stripped, '' if none), state
    (canonical full name, '' if unrecognized), tokens, full (cleaned
    string), is_empty. Deliberately order-agnostic: street/city/state
    reordering is one of the documented corruption types, so nothing here
    depends on token position beyond house-number/state heuristics."""
    if not isinstance(raw, str) or not raw.strip():
        return {"house_number": "", "state": "", "tokens": [], "full": "", "is_empty": True}

    s = romanize_mixed(raw)
    s = _unidecode(s).lower()
    s = _WS_RE.sub(" ", _NONALNUM_RE.sub(" ", s)).strip()
    tokens = s.split()

    # House number: prefer a purely-numeric token (handles "G-1" -> "g","1"
    # correctly even when an earlier unit/suite token like "E-3A" -> "e","3a"
    # appears first); fall back to the first digit-bearing token otherwise.
    house_number = ""
    for tok in tokens:
        if tok.isdigit():
            house_number = tok.lstrip("0") or "0"
            break
    if not house_number:
        for tok in tokens:
            m = _NUM_RE.search(tok)
            if m:
                house_number = m.group(0).lstrip("0") or "0"
                break

    # State: a standalone 2-letter abbreviation is checked first -- it is far
    # less likely to be a coincidental match than a full state name, which
    # can also appear as a street name (e.g. "Washington Street" in NC must
    # not be read as the state of Washington).
    state = ""
    for tok in tokens:
        if tok in STATE_ABBR:
            state = STATE_ABBR[tok]
            break
    if not state:
        joined = " " + s + " "
        for full, canon in FULL_NAME_CANON.items():
            if (" " + full + " ") in joined:
                state = canon
                break

    return {"house_number": house_number, "state": state, "tokens": tokens, "full": s, "is_empty": False}


# ==============================================================================
# SECTION 3 -- BLOCKING
# ==============================================================================
from collections import defaultdict

_SUFFIX_WORDS = set()
for _phrase in LEGAL_SUFFIXES:
    _SUFFIX_WORDS.update(w.strip(".") for w in _phrase.split())

COMMON_TOKEN_DF_FRAC = 0.01
COMMON_TOKEN_DF_MIN = 40
HOPELESS_TOKEN_DF_FRAC = 0.05
HOPELESS_TOKEN_DF_MIN = 300
DEFAULT_MAX_CANDIDATES = 40


class BlockIndex:
    """Inverted index over the target pool (Source-2 + Source-3). Recall-
    oriented by design: no blind posting-count cap (the bug that capped
    V11's recall) -- common tokens are down-weighted (require corroboration)
    or dropped by document frequency, and an oversized union is narrowed by
    cheap pre-ranking (rank_and_cap), never by discarding index entries."""

    def __init__(self):
        self.name_token_idx = defaultdict(list)
        self.prefix_idx = defaultdict(list)
        self.addr_idx = defaultdict(list)
        self.state_only_idx = defaultdict(list)
        self.token_df = defaultdict(int)
        self.pool_size_by_country = defaultdict(int)
        self.n_rows = 0

    def add_record(self, row_pos, country, name_norm, addr_norm):
        self.n_rows += 1
        self.pool_size_by_country[country] += 1
        seen = set()
        for field in ("full", "core", "dba_alt"):
            for tok in (name_norm.get(field) or "").split():
                if len(tok) < 2 or tok in _SUFFIX_WORDS or tok in seen:
                    continue
                key = (country, tok)
                self.name_token_idx[key].append(row_pos)
                self.token_df[key] += 1
                seen.add(tok)

        full_nospace = (name_norm.get("full") or "").replace(" ", "")
        if len(full_nospace) >= 3:
            self.prefix_idx[(country, full_nospace[:5])].append(row_pos)

        # House number + state: state survives corruption far better than
        # street tokens (which get reordered/renamed/abbreviated
        # unpredictably), so it pairs more reliably with house number than a
        # street-name key would. The bucketed key (house number // 10)
        # tolerates single-digit house-number typos (e.g. 9236 vs 9238)
        # without materially widening the block.
        if addr_norm and not addr_norm["is_empty"] and addr_norm["house_number"]:
            hn, state = addr_norm["house_number"], addr_norm["state"]
            if state:
                self.addr_idx[(country, "hn", hn, state)].append(row_pos)
                if hn.isdigit():
                    self.addr_idx[(country, "hnb", str(int(hn) // 10), state)].append(row_pos)
            else:
                key = (country, "hn", hn)
                self.addr_idx[key].append(row_pos)
                self.token_df[key] += 1
        elif addr_norm and not addr_norm["is_empty"] and addr_norm["state"]:
            # No house number at all (common for rural/village-style Indian
            # addresses) -- state alone is the only address signal available.
            self.state_only_idx[(country, addr_norm["state"])].append(row_pos)

    def _tier(self, country, tok):
        pool = self.pool_size_by_country.get(country, 0)
        df = self.token_df.get((country, tok), 0)
        if df > max(HOPELESS_TOKEN_DF_MIN, int(pool * HOPELESS_TOKEN_DF_FRAC)):
            return "hopeless"
        if df > max(COMMON_TOKEN_DF_MIN, int(pool * COMMON_TOKEN_DF_FRAC)):
            return "weak"
        return "strong"

    def candidates(self, country, name_norm, addr_norm):
        """Returns the raw candidate row-position set for one query (before
        pre-rank/cap -- see rank_and_cap)."""
        strong = set()
        weak_hits = defaultdict(int)

        tokens = set()
        for field in ("full", "core", "dba_alt"):
            tokens.update(t for t in (name_norm.get(field) or "").split()
                          if len(t) >= 2 and t not in _SUFFIX_WORDS)

        any_strong_token = False
        for tok in tokens:
            key = (country, tok)
            if key not in self.name_token_idx:
                continue
            tier = self._tier(country, tok)
            if tier == "hopeless":
                continue
            elif tier == "weak":
                for rp in self.name_token_idx[key]:
                    weak_hits[rp] += 1
            else:
                any_strong_token = True
                strong.update(self.name_token_idx[key])

        full_nospace = (name_norm.get("full") or "").replace(" ", "")
        if len(full_nospace) >= 3:
            strong.update(self.prefix_idx.get((country, full_nospace[:5]), []))

        if addr_norm and not addr_norm["is_empty"] and addr_norm["house_number"]:
            hn, state = addr_norm["house_number"], addr_norm["state"]
            if state:
                strong.update(self.addr_idx.get((country, "hn", hn, state), []))
                if hn.isdigit():
                    strong.update(self.addr_idx.get((country, "hnb", str(int(hn) // 10), state), []))
            else:
                key = (country, "hn", hn)
                ceiling = max(HOPELESS_TOKEN_DF_MIN, int(self.pool_size_by_country.get(country, 0) * HOPELESS_TOKEN_DF_FRAC))
                if self.token_df.get(key, 0) <= ceiling:
                    strong.update(self.addr_idx.get(key, []))
        elif addr_norm and not addr_norm["is_empty"] and addr_norm["state"]:
            for rp in self.state_only_idx.get((country, addr_norm["state"]), []):
                weak_hits[rp] += 1

        # Weak hits (common name tokens, or state-only) count if corroborated
        # by another weak hit, or if nothing else was found at all (then
        # it's the best signal available, however thin).
        for rp, cnt in weak_hits.items():
            if cnt >= 2 or not (any_strong_token or strong):
                strong.add(rp)

        return strong


def rank_and_cap(query_full_name, candidate_positions, target_full_names, max_candidates=DEFAULT_MAX_CANDIDATES):
    """Narrows an oversized candidate union with a cheap pre-rank so the
    (slower, richer) feature-scoring stage only ever sees a bounded number
    of candidates per query. This is the direct fix for V11's recall bug:
    that version capped postings AT THE INDEX, dropping entries before any
    ranking happened, so misses were arbitrary. Here, nothing is dropped
    until every candidate has at least been cheaply compared -- a true match
    is only lost if it isn't even a top-N lookalike by name."""
    positions = list(candidate_positions)
    if len(positions) <= max_candidates:
        return positions
    scored = [(fuzz.token_set_ratio(query_full_name, target_full_names[p]), p) for p in positions]
    scored.sort(reverse=True)
    return [p for _, p in scored[:max_candidates]]


# ==============================================================================
# SECTION 4 -- PAIRWISE FEATURES
# ==============================================================================

FEATURE_NAMES = [
    "name_token_sort", "name_token_set", "name_partial", "name_nospace_partial",
    "name_exact_core", "used_alt_name", "name_len_ratio",
    "addr_house_match", "addr_state_match", "addr_token_sort", "addr_token_set",
    "addr_target_empty",
]


def _name_variants(norm):
    variants, seen = [], set()
    for field in ("full", "core", "leet", "dba_alt"):
        v = norm.get(field) or ""
        if v and v not in seen:
            variants.append((field, v))
            seen.add(v)
    return variants or [("full", "")]


def _house_number_similarity(a, b):
    """1.0 exact; 0.8 if one is a prefix/suffix of the other after leading-
    zero stripping (covers both the leading-zero pattern, e.g. 0337 vs 337,
    and the truncation pattern, e.g. 8706 vs 870 or 201 vs 01); else 0."""
    if not a or not b:
        return 0.5
    if a == b:
        return 1.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if longer.endswith(shorter) or longer.startswith(shorter):
        return 0.8
    return 0.0


def name_features(q_norm, t_norm):
    q_variants, t_variants = _name_variants(q_norm), _name_variants(t_norm)
    best = {"token_sort": 0.0, "token_set": 0.0, "partial": 0.0, "nospace_partial": 0.0}
    used_alt = 0
    for qf, qv in q_variants:
        for tf, tv in t_variants:
            if not qv or not tv:
                continue
            ts, tset = fuzz.token_sort_ratio(qv, tv), fuzz.token_set_ratio(qv, tv)
            pr = fuzz.partial_ratio(qv, tv)
            nsp = fuzz.partial_ratio(qv.replace(" ", ""), tv.replace(" ", ""))
            if max(ts, tset, pr, nsp) > max(best.values()):
                used_alt = 1 if (qf == "dba_alt" or tf == "dba_alt") else 0
            best["token_sort"] = max(best["token_sort"], ts)
            best["token_set"] = max(best["token_set"], tset)
            best["partial"] = max(best["partial"], pr)
            best["nospace_partial"] = max(best["nospace_partial"], nsp)

    exact_core = 1.0 if (q_norm.get("core") and q_norm.get("core") == t_norm.get("core")) else 0.0
    qlen, tlen = len(q_norm.get("full") or ""), len(t_norm.get("full") or "")
    len_ratio = (min(qlen, tlen) / max(qlen, tlen)) if max(qlen, tlen) > 0 else 0.0

    return {
        "name_token_sort": best["token_sort"] / 100.0, "name_token_set": best["token_set"] / 100.0,
        "name_partial": best["partial"] / 100.0, "name_nospace_partial": best["nospace_partial"] / 100.0,
        "name_exact_core": exact_core, "used_alt_name": float(used_alt), "name_len_ratio": len_ratio,
    }


def address_features(q_addr, t_addr):
    if t_addr["is_empty"] or q_addr["is_empty"]:
        return {"addr_house_match": 0.5, "addr_state_match": 0.5, "addr_token_sort": 0.5,
                "addr_token_set": 0.5, "addr_target_empty": 1.0 if t_addr["is_empty"] else 0.0}
    house_match = _house_number_similarity(q_addr["house_number"], t_addr["house_number"])
    if q_addr["state"] and t_addr["state"]:
        state_match = 1.0 if q_addr["state"] == t_addr["state"] else 0.0
    else:
        state_match = 0.5
    return {
        "addr_house_match": house_match, "addr_state_match": state_match,
        "addr_token_sort": fuzz.token_sort_ratio(q_addr["full"], t_addr["full"]) / 100.0,
        "addr_token_set": fuzz.token_set_ratio(q_addr["full"], t_addr["full"]) / 100.0,
        "addr_target_empty": 0.0,
    }


def compute_features(q_norm, q_addr, t_norm, t_addr):
    f = {}
    f.update(name_features(q_norm, t_norm))
    f.update(address_features(q_addr, t_addr))
    return f


def feature_vector(f):
    return [f[name] for name in FEATURE_NAMES]


# ==============================================================================
# SECTION 5 -- PER-QUERY DECISION RULE
# ==============================================================================
#
# For a fixed predicted set S of size k with T true positives in it and N
# true matches in total, F_beta's usual three-term denominator collapses to
# just (beta^2 * N + k):
#     F_beta = (1+beta^2) * T / ((1+beta^2)*T + beta^2*(N-T) + (k-T))
#            = (1+beta^2) * T / (beta^2*N + k)
# Plugging the model's own probabilities in as the expectation for both T
# (partial sum) and N (total sum) gives a per-query optimal stopping rule
# that adapts to how many strong candidates exist -- a single global
# threshold cannot simultaneously get 0-match, 1-match and 5-match queries
# right, but this rule handles all three from the same probabilities.

def select_matches(candidate_ids, probs, beta=0.5):
    """Returns the subset of candidate_ids to predict as matches (possibly
    empty), maximizing expected F_beta under the model's own probabilities."""
    if not candidate_ids:
        return []
    order = sorted(range(len(probs)), key=lambda i: -probs[i])
    sorted_probs = [probs[i] for i in order]
    sorted_ids = [candidate_ids[i] for i in order]

    hat_N = sum(sorted_probs)
    hat_p0 = 1.0
    for p in sorted_probs:
        hat_p0 *= (1.0 - p)

    beta2 = beta * beta
    best_k, best_score, cum = 0, hat_p0, 0.0
    for k in range(1, len(sorted_probs) + 1):
        cum += sorted_probs[k - 1]
        score = (1.0 + beta2) * cum / (beta2 * hat_N + k)
        if score > best_score:
            best_score, best_k = score, k

    return sorted_ids[:best_k]


# ==============================================================================
# SECTION 6 -- DATA LOADING / INDEX BUILDING
# ==============================================================================

def load_records(path):
    """Source1/Source2/Source3 records all share the entity_id / business_name
    / business_address / country schema."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def load_ground_truth(path):
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {r["source1_entity_id"]: (set(r["matched_entity_ids"].split(",")) if r["matched_entity_ids"] else set())
            for _, r in gt.iterrows()}


def precompute_targets(tg_df):
    names = [normalize_name(n) for n in tg_df["business_name"]]
    addrs = [normalize_address(a) for a in tg_df["business_address"]]
    ids = tg_df["entity_id"].tolist()
    fulls = [n["full"] for n in names]
    return names, addrs, ids, fulls


def build_index(tg_df, names, addrs):
    idx = BlockIndex()
    for i, country in enumerate(tg_df["country"]):
        idx.add_record(i, country, names[i], addrs[i])
    return idx


# ==============================================================================
# SECTION 7 -- MULTIPROCESSING WORKER + ORCHESTRATION
# ==============================================================================
#
# The target-pool index and precomputed normalizations are set as plain
# module-level state in the main process BEFORE the worker pool is created.
# Under fork (the Linux/Kaggle default), child processes inherit this memory
# copy-on-write at essentially zero cost; passing a multi-GB index through
# Pool's initargs instead would mean pickling and re-sending it once per
# worker. On a spawn-default platform (Windows, some macOS setups) this
# sharing does not happen -- pass --n-jobs 1 there, or run inside WSL/Docker.

_SHARED = {}


def _configure_shared(idx, tg_names, tg_addrs, tg_ids, tg_fulls, max_candidates):
    _SHARED.update(idx=idx, tg_names=tg_names, tg_addrs=tg_addrs, tg_ids=tg_ids,
                    tg_fulls=tg_fulls, max_candidates=max_candidates)


def _process_query_task(args):
    """One query in -> (qid, [(candidate_entity_id, feature_dict), ...]) out."""
    qid, name, address, country = args
    qname = normalize_name(name)
    qaddr = normalize_address(address)
    idx = _SHARED["idx"]
    raw = idx.candidates(country, qname, qaddr)
    capped = rank_and_cap(qname["full"], raw, _SHARED["tg_fulls"], _SHARED["max_candidates"])
    out = []
    for p in capped:
        feats = compute_features(qname, qaddr, _SHARED["tg_names"][p], _SHARED["tg_addrs"][p])
        out.append((_SHARED["tg_ids"][p], feats))
    return qid, out


def _make_pool(n_jobs):
    if not n_jobs or n_jobs < 1:
        n_jobs = os.cpu_count() or 1
    start_method = "fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method()
    ctx = mp.get_context(start_method)
    return ctx.Pool(processes=n_jobs), n_jobs


def _run_tasks(tasks, n_jobs):
    """Yields (qid, pairs) for every task, in parallel if n_jobs > 1."""
    if n_jobs == 1:
        for t in tasks:
            yield _process_query_task(t)
        return
    pool, n_jobs = _make_pool(n_jobs)
    chunksize = max(1, len(tasks) // (n_jobs * 8))
    try:
        for qid, pairs in pool.imap_unordered(_process_query_task, tasks, chunksize=chunksize):
            yield qid, pairs
    finally:
        pool.close()
        pool.join()


def _macro_f05(qid_list, scored_df, gt_map, beta=0.5):
    if len(qid_list) == 0:
        return 0.0, 0
    beta2 = beta * beta
    by_q = scored_df.groupby("qid") if len(scored_df) else None
    scores = []
    for qid in qid_list:
        true_ids = gt_map.get(qid, set())
        chosen = set()
        if by_q is not None and qid in by_q.groups:
            g = by_q.get_group(qid)
            chosen = set(select_matches(g["cand_id"].tolist(), g["prob"].tolist(), beta=beta))
        tp, fp, fn = len(chosen & true_ids), len(chosen - true_ids), len(true_ids - chosen)
        if tp == 0 and fp == 0:
            f = 1.0 if fn == 0 else 0.0
        else:
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f = (1 + beta2) * prec * rec / (beta2 * prec + rec) if (prec + rec) > 0 else 0.0
        scores.append(f)
    return sum(scores) / len(scores), len(scores)


def _log(t0, msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", file=sys.stderr)


def run_train(args):
    if joblib is None:
        sys.exit("joblib is required for training (pip install joblib)")
    import lightgbm as lgb

    t0 = time.time()
    s1 = load_records(args.source1)
    tg = load_records(args.targets)
    gt_map = load_ground_truth(args.ground_truth)
    _log(t0, f"loaded {len(s1)} queries, {len(tg)} targets, {len(gt_map)} ground-truth rows")

    tg_names, tg_addrs, tg_ids, tg_fulls = precompute_targets(tg)
    idx = build_index(tg, tg_names, tg_addrs)
    _log(t0, f"index built ({idx.n_rows} records)")
    _configure_shared(idx, tg_names, tg_addrs, tg_ids, tg_fulls, args.max_candidates)

    tasks = list(zip(s1["entity_id"], s1["business_name"], s1["business_address"], s1["country"]))
    rows = []
    n_done = 0
    for qid, pairs in _run_tasks(tasks, args.n_jobs):
        true_ids = gt_map.get(qid, set())
        for cand_id, feats in pairs:
            rows.append({"qid": qid, "cand_id": cand_id, "label": 1 if cand_id in true_ids else 0, **feats})
        n_done += 1
        if n_done % 200000 == 0:
            _log(t0, f"  ...{n_done}/{len(tasks)} queries processed")
    _log(t0, f"built {len(rows)} training pairs ({sum(r['label'] for r in rows)} positive)")

    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("No candidate pairs were generated at all -- check that source1/targets share a country vocabulary.")

    rng = np.random.RandomState(42)
    uniq_q = list(df["qid"].unique())
    rng.shuffle(uniq_q)
    val_q = set(uniq_q[: max(1, int(len(uniq_q) * args.val_frac))])
    is_val = df["qid"].isin(val_q)

    X, y = df[FEATURE_NAMES], df["label"]
    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="binary",
        random_state=42, verbosity=-1, n_jobs=max(1, os.cpu_count() or 1),
    )
    model.fit(X[~is_val], y[~is_val], eval_set=[(X[is_val], y[is_val])],
              callbacks=[lgb.early_stopping(30, verbose=False)])
    _log(t0, f"model trained, best_iteration={model.best_iteration_}")

    val_df = df[is_val].copy()
    val_df["prob"] = model.predict_proba(val_df[FEATURE_NAMES])[:, 1]
    macro, n = _macro_f05(sorted(val_q), val_df, gt_map)
    _log(t0, f"end-to-end validation macro F0.5 = {macro:.4f}  (n={n} queries)")
    _log(t0, "NOTE: this number reflects only the queries/negatives present in the "
              "training sample provided -- validate again once trained on the full "
              "training set, where the candidate pool is ~10.3M records rather than "
              "whatever --targets contained here.")

    joblib.dump({"model": model, "max_candidates": args.max_candidates}, args.model_out)
    _log(t0, f"model saved to {args.model_out}")


def run_predict(args):
    if joblib is None:
        sys.exit("joblib is required to load the trained model (pip install joblib)")

    t0 = time.time()
    bundle = joblib.load(args.model)
    model = bundle["model"]
    max_candidates = args.max_candidates or bundle.get("max_candidates", DEFAULT_MAX_CANDIDATES)

    s1 = load_records(args.source1)
    tg = load_records(args.targets)
    _log(t0, f"loaded {len(s1)} queries, {len(tg)} targets")

    tg_names, tg_addrs, tg_ids, tg_fulls = precompute_targets(tg)
    idx = build_index(tg, tg_names, tg_addrs)
    _log(t0, f"index built ({idx.n_rows} records)")
    _configure_shared(idx, tg_names, tg_addrs, tg_ids, tg_fulls, max_candidates)

    tasks = list(zip(s1["entity_id"], s1["business_name"], s1["business_address"], s1["country"]))
    match_rows, cand_rows = [], []
    n_done = 0
    for qid, pairs in _run_tasks(tasks, args.n_jobs):
        if not pairs:
            match_rows.append((qid, ""))
            cand_rows.append((qid, ""))
        else:
            cand_ids = [cid for cid, _ in pairs]
            X = pd.DataFrame([f for _, f in pairs], columns=FEATURE_NAMES)
            probs = model.predict_proba(X)[:, 1].tolist()
            chosen = select_matches(cand_ids, probs, beta=0.5)
            match_rows.append((qid, ",".join(chosen)))
            cand_rows.append((qid, ",".join(cand_ids)))
        n_done += 1
        if n_done % 200000 == 0:
            _log(t0, f"  ...{n_done}/{len(tasks)} queries scored")
    _log(t0, "all queries scored")

    os.makedirs(args.output_dir, exist_ok=True)
    match_path = os.path.join(args.output_dir, "matching_results.tsv")
    cand_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    pd.DataFrame(match_rows, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(
        match_path, sep="\t", index=False)
    pd.DataFrame(cand_rows, columns=["source1_entity_id", "candidate_entity_ids"]).to_csv(
        cand_path, sep="\t", index=False)
    n_singleton = sum(1 for _, m in match_rows if m == "")
    _log(t0, f"wrote {match_path} and {cand_path}  ({n_singleton}/{len(match_rows)} predicted singletons)")


# ==============================================================================
# SECTION 8 -- CLI
# ==============================================================================

def main():
    p = argparse.ArgumentParser(description="Amazon ML Challenge 2026 entity-resolution pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train", help="Train the pairwise classifier and save it")
    pt.add_argument("--source1", required=True)
    pt.add_argument("--targets", required=True)
    pt.add_argument("--ground-truth", required=True)
    pt.add_argument("--model-out", default="model.pkl")
    pt.add_argument("--val-frac", type=float, default=0.15)
    pt.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    pt.add_argument("--n-jobs", type=int, default=None, help="default: all cores")
    pt.set_defaults(func=run_train)

    pp = sub.add_parser("predict", help="Score queries and write the two output files")
    pp.add_argument("--source1", required=True)
    pp.add_argument("--targets", required=True)
    pp.add_argument("--model", required=True)
    pp.add_argument("--output-dir", default="./output")
    pp.add_argument("--max-candidates", type=int, default=None, help="default: whatever was used in training")
    pp.add_argument("--n-jobs", type=int, default=None, help="default: all cores")
    pp.set_defaults(func=run_predict)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
