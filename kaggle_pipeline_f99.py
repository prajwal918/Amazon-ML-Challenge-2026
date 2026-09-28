#!/usr/bin/env python3
"""
================================================================================
Amazon ML Challenge 2026 -- Production Grandmaster Pipeline (F99 Target)
================================================================================
Multi-source entity resolution pipeline for Kaggle Cloud & Local execution.
Resolves 1,732,544 queries from test_source1.tsv against ~10M records across
test_source2.tsv and test_source3.tsv (US, India, and France).
Optimizes Macro-Averaged F0.5.
================================================================================
"""

import os
import sys
import re
import time
import zipfile
import argparse
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode as _unidecode
import joblib
import lightgbm as lgb

# Indic transliteration support
_HAVE_INDIC = False
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

# ==============================================================================
# SECTION 1 -- NORMALIZATION
# ==============================================================================

LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "pvt", "llc", "l.l.c",
    "incorporated", "inc", "corp", "corporation", "services", "center",
    "centre", "holdings", "group", "enterprises", "solutions", "llp",
    "l.l.p", "sarl", "sas", "ltd", "limited", "co", "company", "pllc", "plc",
    "gmbh", "sasu", "eurl", "sa", "praiveta limiteda", "praivett limitted"
]
_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(s) for s in LEGAL_SUFFIXES), key=len, reverse=True)) + r")\.?\b",
    re.IGNORECASE,
)
_METADATA_RE = re.compile(r"\(\s*id\s*[:#]?\s*\d+\s*\)", re.IGNORECASE)
_HANDLE_RE = re.compile(r"(?<![a-z0-9])@")
_DOMAIN_RE = re.compile(r"\.(com|net|org|io|co|in|fr|biz)\b", re.IGNORECASE)
_DBA_RE = re.compile(r"\bd\.?\s*b\.?\s*a\.?\s*:?\s*", re.IGNORECASE)
_LEET_MAP = str.maketrans({"5": "s", "3": "e", "1": "i", "0": "o", "4": "a", "7": "t", "8": "b"})
_NONALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")

def _clean_basic(s):
    return _WS_RE.sub(" ", _NONALNUM_RE.sub(" ", s)).strip()

def normalize_name(raw):
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
        cand = core.translate(_LEET_MAP)
        if cand != core:
            leet = cand
    return {"full": full, "core": core, "leet": leet, "dba_alt": dba_alt}

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
STATE_ABBR = {**US_STATES, **IN_STATES}
FULL_TO_ABBR = {v: k for k, v in STATE_ABBR.items()}
FULL_NAME_CANON = {full: full for full in FULL_TO_ABBR}
FULL_NAME_CANON.update({"orissa": "odisha", "uttaranchal": "uttarakhand", "pondicherry": "puducherry"})

_NUM_RE = re.compile(r"\d+")
_POSTAL_US_FR = re.compile(r"\b\d{5}\b")
_POSTAL_IN = re.compile(r"\b\d{6}\b")

def normalize_address(raw, country=""):
    if not isinstance(raw, str) or not raw.strip():
        return {"house_number": "", "state": "", "postal": "", "tokens": [], "full": "", "is_empty": True}
    s = romanize_mixed(raw)
    s = _unidecode(s).lower()
    s = _WS_RE.sub(" ", _NONALNUM_RE.sub(" ", s)).strip()
    tokens = s.split()

    # Postal code extraction
    postal = ""
    if country == "India":
        m = _POSTAL_IN.search(s)
        if m: postal = m.group(0)
    else:
        m = _POSTAL_US_FR.search(s)
        if m: postal = m.group(0)

    # House number: prefer purely-numeric token distinct from postal
    house_number = ""
    for tok in tokens:
        if tok.isdigit() and tok != postal and len(tok) <= 5:
            house_number = tok.lstrip("0") or "0"
            break
    if not house_number:
        for tok in tokens:
            m = _NUM_RE.search(tok)
            if m:
                val = m.group(0)
                if val != postal and len(val) <= 5:
                    house_number = val.lstrip("0") or "0"
                    break

    # State: 2-letter abbreviation or full name
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

    return {"house_number": house_number, "state": state, "postal": postal, "tokens": tokens, "full": s, "is_empty": False}

# ==============================================================================
# SECTION 2 -- BLOCKING INDEX
# ==============================================================================

_SUFFIX_WORDS = set()
for _phrase in LEGAL_SUFFIXES:
    _SUFFIX_WORDS.update(w.strip(".") for w in _phrase.split())

COMMON_TOKEN_DF_FRAC = 0.01
COMMON_TOKEN_DF_MIN = 40
HOPELESS_TOKEN_DF_FRAC = 0.05
HOPELESS_TOKEN_DF_MIN = 300
DEFAULT_MAX_CANDIDATES = 25

class BlockIndex:
    def __init__(self):
        self.name_token_idx = defaultdict(list)
        self.prefix_idx = defaultdict(list)
        self.addr_idx = defaultdict(list)
        self.postal_idx = defaultdict(list)
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

        if addr_norm and not addr_norm["is_empty"]:
            hn, state, postal = addr_norm["house_number"], addr_norm["state"], addr_norm["postal"]
            if postal:
                self.postal_idx[(country, postal)].append(row_pos)
            if hn:
                if state:
                    self.addr_idx[(country, "hn", hn, state)].append(row_pos)
                    if hn.isdigit():
                        self.addr_idx[(country, "hnb", str(int(hn) // 10), state)].append(row_pos)
                else:
                    key = (country, "hn", hn)
                    self.addr_idx[key].append(row_pos)
                    self.token_df[key] += 1
            elif state:
                self.state_only_idx[(country, state)].append(row_pos)

    def _tier(self, country, tok):
        pool = self.pool_size_by_country.get(country, 0)
        df = self.token_df.get((country, tok), 0)
        if df > max(HOPELESS_TOKEN_DF_MIN, int(pool * HOPELESS_TOKEN_DF_FRAC)):
            return "hopeless"
        if df > max(COMMON_TOKEN_DF_MIN, int(pool * COMMON_TOKEN_DF_FRAC)):
            return "weak"
        return "strong"

    def candidates(self, country, name_norm, addr_norm):
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

        if addr_norm and not addr_norm["is_empty"]:
            hn, state, postal = addr_norm["house_number"], addr_norm["state"], addr_norm["postal"]
            if postal and (country, postal) in self.postal_idx:
                p_cands = self.postal_idx[(country, postal)]
                if len(p_cands) <= 300:
                    for rp in p_cands: weak_hits[rp] += 1
            if hn:
                if state:
                    strong.update(self.addr_idx.get((country, "hn", hn, state), []))
                    if hn.isdigit():
                        strong.update(self.addr_idx.get((country, "hnb", str(int(hn) // 10), state), []))
                else:
                    key = (country, "hn", hn)
                    ceiling = max(HOPELESS_TOKEN_DF_MIN, int(self.pool_size_by_country.get(country, 0) * HOPELESS_TOKEN_DF_FRAC))
                    if self.token_df.get(key, 0) <= ceiling:
                        strong.update(self.addr_idx.get(key, []))
            elif state:
                for rp in self.state_only_idx.get((country, state), []):
                    weak_hits[rp] += 1

        for rp, cnt in weak_hits.items():
            if cnt >= 2 or not (any_strong_token or strong):
                strong.add(rp)

        return strong

def rank_and_cap(query_full_name, candidate_positions, target_full_names, max_candidates=DEFAULT_MAX_CANDIDATES):
    positions = list(candidate_positions)
    if len(positions) <= max_candidates:
        return positions
    scored = [(fuzz.token_set_ratio(query_full_name, target_full_names[p]), p) for p in positions]
    scored.sort(reverse=True)
    return [p for _, p in scored[:max_candidates]]

# ==============================================================================
# SECTION 3 -- PAIRWISE FEATURES
# ==============================================================================

FEATURE_NAMES = [
    "name_token_sort", "name_token_set", "name_partial", "name_nospace_partial",
    "name_exact_core", "used_alt_name", "name_len_ratio",
    "addr_house_match", "addr_state_match", "addr_postal_match",
    "addr_token_sort", "addr_token_set", "addr_target_empty",
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
        return {
            "addr_house_match": 0.5, "addr_state_match": 0.5, "addr_postal_match": 0.5,
            "addr_token_sort": 0.5, "addr_token_set": 0.5, "addr_target_empty": 1.0 if t_addr["is_empty"] else 0.0
        }
    house_match = _house_number_similarity(q_addr["house_number"], t_addr["house_number"])
    if q_addr["state"] and t_addr["state"]:
        state_match = 1.0 if q_addr["state"] == t_addr["state"] else 0.0
    else:
        state_match = 0.5

    if q_addr["postal"] and t_addr["postal"]:
        postal_match = 1.0 if q_addr["postal"] == t_addr["postal"] else 0.0
    else:
        postal_match = 0.5

    return {
        "addr_house_match": house_match, "addr_state_match": state_match,
        "addr_postal_match": postal_match,
        "addr_token_sort": fuzz.token_sort_ratio(q_addr["full"], t_addr["full"]) / 100.0,
        "addr_token_set": fuzz.token_set_ratio(q_addr["full"], t_addr["full"]) / 100.0,
        "addr_target_empty": 0.0,
    }

def compute_features(q_norm, q_addr, t_norm, t_addr):
    f = {}
    f.update(name_features(q_norm, t_norm))
    f.update(address_features(q_addr, t_addr))
    return f

# ==============================================================================
# SECTION 4 -- DECISION RULE (MAX EXPECTED F0.5 + SIBLING CORROBORATION)
# ==============================================================================

def select_matches(candidate_ids, probs, beta=0.5, sibling_thresh=0.45):
    if not candidate_ids or not probs:
        return []
    order = np.argsort(probs)[::-1]
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

    chosen = set(sorted_ids[:best_k])
    # Sibling corroboration: if confident match found, include siblings with prob >= sibling_thresh
    if best_k > 0 and sorted_probs[0] >= 0.70:
        for cid, p in zip(sorted_ids, sorted_probs):
            if p >= sibling_thresh:
                chosen.add(cid)

    return [cid for cid in sorted_ids if cid in chosen]

# ==============================================================================
# SECTION 5 -- MULTIPROCESSING ORCHESTRATION
# ==============================================================================

_SHARED = {}

def _configure_shared(idx, tg_names, tg_addrs, tg_ids, tg_fulls, max_candidates):
    _SHARED.update(idx=idx, tg_names=tg_names, tg_addrs=tg_addrs, tg_ids=tg_ids,
                   tg_fulls=tg_fulls, max_candidates=max_candidates)

def _process_query_task(args):
    qid, name, address, country = args
    qname = normalize_name(name)
    qaddr = normalize_address(address, country)
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

def _log(t0, msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

def load_records(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

def load_targets(paths):
    if isinstance(paths, str):
        paths = [p.strip() for p in paths.split(",") if p.strip()]
    dfs = [load_records(p) for p in paths]
    return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

def precompute_targets(tg_df):
    names = [normalize_name(n) for n in tg_df["business_name"]]
    countries = tg_df["country"].tolist()
    addrs = [normalize_address(a, c) for a, c in zip(tg_df["business_address"], countries)]
    ids = tg_df["entity_id"].tolist()
    fulls = [n["full"] for n in names]
    return names, addrs, ids, fulls

def build_index(tg_df, names, addrs):
    idx = BlockIndex()
    for i, country in enumerate(tg_df["country"]):
        idx.add_record(i, country, names[i], addrs[i])
    return idx

# ==============================================================================
# SECTION 6 -- MAIN RUNNER
# ==============================================================================

def run_pipeline(source1_path, target_paths, model_path, output_dir, max_candidates=DEFAULT_MAX_CANDIDATES, n_jobs=None):
    t0 = time.time()
    _log(t0, f"Loading model from {model_path}...")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    
    _log(t0, f"Loading Source 1 queries from {source1_path}...")
    s1 = load_records(source1_path)
    
    _log(t0, f"Loading Target records from {target_paths}...")
    tg = load_targets(target_paths)
    _log(t0, f"Loaded {len(s1):,} queries and {len(tg):,} targets.")
    
    _log(t0, "Precomputing target normalizations...")
    tg_names, tg_addrs, tg_ids, tg_fulls = precompute_targets(tg)
    
    _log(t0, "Building inverted block index...")
    idx = build_index(tg, tg_names, tg_addrs)
    _log(t0, f"Block index ready with {idx.n_rows:,} records.")
    
    _configure_shared(idx, tg_names, tg_addrs, tg_ids, tg_fulls, max_candidates)
    
    tasks = list(zip(s1["entity_id"], s1["business_name"], s1["business_address"], s1["country"]))
    _log(t0, f"Starting candidate generation and scoring across {n_jobs or os.cpu_count()} CPU cores...")
    
    match_dict = {}
    cand_dict = {}
    
    # Process queries in batches for ultra-fast vectorized LightGBM prediction
    batch_qids = []
    batch_cids = []
    batch_feats = []
    
    n_done = 0
    t_last = time.time()
    
    def flush_batch():
        if not batch_qids: return
        all_feats = []
        counts = []
        for flist in batch_feats:
            counts.append(len(flist))
            all_feats.extend(flist)
        
        if all_feats:
            X = pd.DataFrame(all_feats, columns=FEATURE_NAMES)
            probs = model.predict_proba(X)[:, 1]
            
            offset = 0
            for i, qid in enumerate(batch_qids):
                cnt = counts[i]
                cids = batch_cids[i]
                if cnt == 0:
                    match_dict[qid] = ""
                    cand_dict[qid] = ""
                else:
                    q_probs = probs[offset:offset+cnt].tolist()
                    chosen = select_matches(cids, q_probs, beta=0.5, sibling_thresh=0.45)
                    match_dict[qid] = ",".join(chosen)
                    cand_dict[qid] = ",".join(cids)
                    offset += cnt
        else:
            for qid in batch_qids:
                match_dict[qid] = ""
                cand_dict[qid] = ""
                
        batch_qids.clear()
        batch_cids.clear()
        batch_feats.clear()

    for qid, pairs in _run_tasks(tasks, n_jobs):
        batch_qids.append(qid)
        cids = [cid for cid, _ in pairs]
        batch_cids.append(cids)
        batch_feats.append([f for _, f in pairs])
        
        if len(batch_qids) >= 10000:
            flush_batch()
            
        n_done += 1
        if n_done % 100000 == 0:
            now = time.time()
            rate = 100000 / (now - t_last)
            _log(t0, f"  Scored {n_done:,} / {len(tasks):,} queries ({rate:.0f} q/s)")
            t_last = now
            
    flush_batch()
    _log(t0, f"All {len(tasks):,} queries successfully scored!")
    
    os.makedirs(output_dir, exist_ok=True)
    match_file = os.path.join(output_dir, "matching_results.tsv")
    cand_file = os.path.join(output_dir, "candidate_pairs.tsv")
    
    _log(t0, f"Writing {match_file}...")
    with open(match_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for qid in s1["entity_id"]:
            f.write(f"{qid}\t{match_dict.get(qid, '')}\n")
            
    _log(t0, f"Writing {cand_file}...")
    with open(cand_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for qid in s1["entity_id"]:
            f.write(f"{qid}\t{cand_dict.get(qid, '')}\n")
            
    n_singletons = sum(1 for m in match_dict.values() if not m)
    _log(t0, f"Output stats: {len(s1):,} rows, {n_singletons:,} singletons ({n_singletons/len(s1)*100:.2f}%)")
    
    # Create submission.zip
    zip_path = os.path.join(output_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(match_file, "matching_results.tsv")
        zf.write(cand_file, "candidate_pairs.tsv")
    _log(t0, f"Created {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.1f} MB)")
    _log(t0, "Pipeline completed successfully!")

def load_ground_truth(path):
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {r["source1_entity_id"]: (set(r["matched_entity_ids"].split(",")) if r["matched_entity_ids"] else set())
            for _, r in gt.iterrows()}

def run_train(source1_path, target_paths, ground_truth_path, model_out="model_f99.pkl", val_frac=0.15, max_candidates=DEFAULT_MAX_CANDIDATES, n_jobs=None):
    t0 = time.time()
    s1 = load_records(source1_path)
    tg = load_targets(target_paths)
    gt_map = load_ground_truth(ground_truth_path)
    _log(t0, f"Loaded {len(s1):,} queries, {len(tg):,} targets, {len(gt_map):,} ground truth rows.")
    
    _log(t0, "Precomputing target normalizations...")
    tg_names, tg_addrs, tg_ids, tg_fulls = precompute_targets(tg)
    
    _log(t0, "Building inverted block index...")
    idx = build_index(tg, tg_names, tg_addrs)
    _log(t0, f"Block index ready ({idx.n_rows:,} records)")
    
    _configure_shared(idx, tg_names, tg_addrs, tg_ids, tg_fulls, max_candidates)
    
    tasks = list(zip(s1["entity_id"], s1["business_name"], s1["business_address"], s1["country"]))
    _log(t0, f"Generating candidate pairs for training across {n_jobs or 1} CPUs...")
    rows = []
    n_done = 0
    for qid, pairs in _run_tasks(tasks, n_jobs):
        true_ids = gt_map.get(qid, set())
        for cand_id, feats in pairs:
            rows.append({"qid": qid, "cand_id": cand_id, "label": 1 if cand_id in true_ids else 0, **feats})
        n_done += 1
        
    _log(t0, f"Built {len(rows):,} training pairs ({sum(r['label'] for r in rows):,} positive).")
    df = pd.DataFrame(rows)
    
    rng = np.random.RandomState(42)
    uniq_q = list(df["qid"].unique())
    rng.shuffle(uniq_q)
    val_q = set(uniq_q[: max(1, int(len(uniq_q) * val_frac))])
    is_val = df["qid"].isin(val_q)
    
    X, y = df[FEATURE_NAMES], df["label"]
    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="binary",
        random_state=42, verbosity=-1, n_jobs=max(1, os.cpu_count() or 1),
    )
    model.fit(X[~is_val], y[~is_val], eval_set=[(X[is_val], y[is_val])],
              callbacks=[lgb.early_stopping(30, verbose=False)])
    _log(t0, f"Model trained! Best iteration = {model.best_iteration_}")
    
    joblib.dump({"model": model, "max_candidates": max_candidates}, model_out)
    _log(t0, f"Model saved to {model_out}")
    return model_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    
    pt = sub.add_parser("train")
    pt.add_argument("--source1", required=True)
    pt.add_argument("--targets", required=True)
    pt.add_argument("--ground-truth", required=True)
    pt.add_argument("--model-out", default="model_f99.pkl")
    pt.add_argument("--val-frac", type=float, default=0.15)
    pt.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    pt.add_argument("--n-jobs", type=int, default=None)
    
    pp = sub.add_parser("predict")
    pp.add_argument("--source1", required=True)
    pp.add_argument("--targets", required=True)
    pp.add_argument("--model", required=True)
    pp.add_argument("--output-dir", default="./output")
    pp.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    pp.add_argument("--n-jobs", type=int, default=None)
    
    args = parser.parse_args()
    if args.command == "train":
        run_train(args.source1, args.targets, args.ground_truth, args.model_out, args.val_frac, args.max_candidates, args.n_jobs)
    elif args.command == "predict":
        run_pipeline(args.source1, args.targets, args.model, args.output_dir, args.max_candidates, args.n_jobs)
    else:
        parser.print_help()

