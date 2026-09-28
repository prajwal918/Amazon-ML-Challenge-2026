#!/usr/bin/env python3
"""
================================================================================
Amazon ML Challenge 2026 -- Production Batch Inference Pipeline V19
================================================================================
Batch-scores candidate_pairs.tsv with the finetuned 27-feature LightGBM model.
Optimized for 20-Core CPU with low-memory streaming (< 1.5 GB RAM).
================================================================================
"""

import sys
import os
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from unidecode import unidecode as _unidecode
import lightgbm as lgb
import joblib

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

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

LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt. ltd.", "pvt", "llc", "l.l.c",
    "incorporated", "inc", "corp", "corporation", "services", "center",
    "centre", "holdings", "group", "enterprises", "solutions", "llp",
    "l.l.p", "sarl", "sas", "ltd", "limited", "co", "company", "pllc", "plc",
    "gmbh", "sasu", "eurl", "sa", "praiveta limiteda", "praivett limitted",
    "industries", "associates", "consulting", "international", "intl"
]
_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(s) for s in LEGAL_SUFFIXES), key=len, reverse=True)) + r")\.?\b",
    re.IGNORECASE,
)
_METADATA_RE = re.compile(r"\(\s*id\s*[:#]?\s*\d+\s*\)", re.IGNORECASE)
_HANDLE_RE = re.compile(r"(?<![a-z0-9])@")
_DOMAIN_RE = re.compile(r"\.(com|net|org|io|co|in|fr|biz|org\.in|gov\.in)\b", re.IGNORECASE)
_DBA_RE = re.compile(r"\bd\.?\s*b\.?\s*a\.?\s*:?\s*", re.IGNORECASE)
_LEET_MAP = str.maketrans({"5": "s", "3": "e", "1": "i", "0": "o", "4": "a", "7": "t", "8": "b"})
_NONALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")
_STOPWORDS = {"the", "and", "of", "a", "an", "in", "to", "for", "at", "on", "by", "null", "near", "opp", "no"}

def _clean_basic(s):
    return _WS_RE.sub(" ", _NONALNUM_RE.sub(" ", s)).strip()

def normalize_name(raw):
    if not isinstance(raw, str) or not raw.strip():
        return {"full": "", "core": "", "comp": "", "leet": "", "dba_alt": "", "tokens": frozenset(), "qgrams": frozenset()}
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
    comp = re.sub(r'[^a-z0-9]', '', core)
    
    leet = ""
    if any(ch.isdigit() for ch in core):
        cand = core.translate(_LEET_MAP)
        if cand != core:
            leet = cand
            
    tokens = frozenset(t for t in core.split() if t not in _STOPWORDS and len(t) > 1)
    qgrams = frozenset(core[i:i+3] for i in range(len(core) - 2)) if len(core) >= 3 else frozenset()
    
    return {"full": full, "core": core, "comp": comp, "leet": leet, "dba_alt": dba_alt, "tokens": tokens, "qgrams": qgrams}

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
    "tg": "telangana", "ts": "telangana", "tr": "tripura", "up": "uttar pradesh", "uk": "uttarakhand",
    "wb": "west bengal", "dl": "delhi",
}
STATE_ABBR = {**US_STATES, **IN_STATES}
FULL_TO_ABBR = {v: k for k, v in STATE_ABBR.items()}
FULL_NAME_CANON = {full: full for full in FULL_TO_ABBR}
FULL_NAME_CANON.update({"orissa": "odisha", "uttaranchal": "uttarakhand", "pondicherry": "puducherry", "tamilnadu": "tamil nadu"})

_NUM_RE = re.compile(r"\d+")
_POSTAL_US_FR = re.compile(r"\b\d{5}\b")
_POSTAL_IN = re.compile(r"\b[1-9]\d{5}\b")

def normalize_address(raw, country=""):
    if not isinstance(raw, str) or not raw.strip():
        return {"house_number": "", "state": "", "postal": "", "tokens": frozenset(), "full": "", "is_empty": True}
    s = romanize_mixed(raw)
    s = _unidecode(s).lower()
    s = _WS_RE.sub(" ", _NONALNUM_RE.sub(" ", s)).strip()
    tokens = s.split()

    postal = ""
    if country == "India":
        m = _POSTAL_IN.search(s)
        if m: postal = m.group(0)
    else:
        m = _POSTAL_US_FR.search(s)
        if m: postal = m.group(0)

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

    state = ""
    cleaned_s = re.sub(r'\bfl\b\.?\s*(?:floor|ground|\d)', ' ', s)
    s_tokens = cleaned_s.split()
    for tok in s_tokens:
        if tok in STATE_ABBR:
            state = STATE_ABBR[tok]
            break
    if not state:
        joined = " " + s + " "
        for full, canon in FULL_NAME_CANON.items():
            if (" " + full + " ") in joined:
                state = canon
                break

    tok_set = frozenset(t for t in tokens if t not in _STOPWORDS and len(t) > 1)
    return {"house_number": house_number, "state": state, "postal": postal, "tokens": tok_set, "full": s, "is_empty": False}

def compute_pair_features(q_norm, q_addr, t_norm, t_addr):
    q_full, t_full = q_norm["full"], t_norm["full"]
    q_core, t_core = q_norm["core"], t_norm["core"]
    
    tsort = fuzz.token_sort_ratio(q_full, t_full) / 100.0
    tset = fuzz.token_set_ratio(q_full, t_full) / 100.0
    partial = fuzz.partial_ratio(q_full, t_full) / 100.0
    
    jw = float(JaroWinkler.similarity(q_full, t_full))
    core_jw = float(JaroWinkler.similarity(q_core, t_core)) if q_core and t_core else 0.0
    core_sort = fuzz.token_sort_ratio(q_core, t_core) / 100.0 if q_core and t_core else 0.0
    core_set = fuzz.token_set_ratio(q_core, t_core) / 100.0 if q_core and t_core else 0.0
    
    exact_core = 1.0 if (q_core and t_core and q_core == t_core) else 0.0
    exact_comp = 1.0 if (q_norm["comp"] and t_norm["comp"] and q_norm["comp"] == t_norm["comp"]) else 0.0
    
    qlen, tlen = len(q_full), len(t_full)
    len_ratio = (min(qlen, tlen) / max(qlen, tlen)) if max(qlen, tlen) > 0 else 0.0
    len_diff = abs(qlen - tlen)
    
    inter_tok = len(q_norm["tokens"] & t_norm["tokens"])
    union_tok = len(q_norm["tokens"] | t_norm["tokens"])
    tok_jaccard = inter_tok / union_tok if union_tok > 0 else 0.0
    
    inter_qg = len(q_norm["qgrams"] & t_norm["qgrams"])
    union_qg = len(q_norm["qgrams"] | t_norm["qgrams"])
    qgram_jaccard = inter_qg / union_qg if union_qg > 0 else 0.0
    
    used_alt = 1.0 if (q_norm["dba_alt"] or t_norm["dba_alt"]) else 0.0
    
    if t_addr["is_empty"] or q_addr["is_empty"]:
        house_match = 0.5
        house_conflict = 0.0
        state_match = 0.5
        state_conflict = 0.0
        postal_match = 0.5
        postal_conflict = 0.0
        addr_sort = 0.5
        addr_set = 0.5
        addr_jacc = 0.5
        target_empty = 1.0 if t_addr["is_empty"] else 0.0
    else:
        target_empty = 0.0
        qh, th = q_addr["house_number"], t_addr["house_number"]
        if qh and th:
            if qh == th:
                house_match = 1.0
                house_conflict = 0.0
            elif qh.startswith(th) or th.startswith(qh) or qh.endswith(th) or th.endswith(qh):
                house_match = 0.8
                house_conflict = 0.0
            else:
                house_match = 0.0
                house_conflict = 1.0
        else:
            house_match = 0.5
            house_conflict = 0.0
            
        qs, ts = q_addr["state"], t_addr["state"]
        if qs and ts:
            if qs == ts:
                state_match = 1.0
                state_conflict = 0.0
            else:
                state_match = 0.0
                state_conflict = 1.0
        else:
            state_match = 0.5
            state_conflict = 0.0
            
        qp, tp = q_addr["postal"], t_addr["postal"]
        if qp and tp:
            if qp == tp:
                postal_match = 1.0
                postal_conflict = 0.0
            else:
                postal_match = 0.0
                postal_conflict = 1.0
        else:
            postal_match = 0.5
            postal_conflict = 0.0
            
        addr_sort = fuzz.token_sort_ratio(q_addr["full"], t_addr["full"]) / 100.0
        addr_set = fuzz.token_set_ratio(q_addr["full"], t_addr["full"]) / 100.0
        
        ainter = len(q_addr["tokens"] & t_addr["tokens"])
        aunion = len(q_addr["tokens"] | t_addr["tokens"])
        addr_jacc = ainter / aunion if aunion > 0 else 0.0

    name_x_addr = tsort * addr_sort
    core_x_addr_set = core_set * addr_set
    
    return [
        tsort, tset, partial,
        jw, core_jw, core_sort, core_set,
        exact_core, exact_comp, len_ratio, float(len_diff),
        tok_jaccard, float(inter_tok), qgram_jaccard, used_alt,
        house_match, house_conflict,
        state_match, state_conflict,
        postal_match, postal_conflict,
        addr_sort, addr_set, addr_jacc,
        target_empty, name_x_addr, core_x_addr_set
    ]

def main():
    t0 = time.time()
    print("=" * 80)
    print("BATCH INFERENCE PIPELINE V19 (STREAMING 20-CORE CPU)")
    print("=" * 80)
    
    model_path = "model_finetuned_v19.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return
        
    bundle = joblib.load(model_path)
    model = bundle["model"]
    base_th = bundle["base_threshold"]
    sib_th = bundle["sibling_threshold"]
    print(f"Loaded model: BaseThreshold={base_th:.2f}, SiblingThreshold={sib_th:.2f}")
    
    BASE_DIR = Path("student_resource/dataset/test")
    CAND_FILE = Path("output/candidate_pairs.tsv")
    OUT_MATCH_FILE = Path("output/matching_results_v19.tsv")
    FINAL_MATCH_FILE = Path("output/matching_results.tsv")
    
    print(f"[1/4] Reading candidate pairs index from {CAND_FILE}...")
    cand_dict = {}
    needed_tgts = set()
    s1_order = []
    
    with open(CAND_FILE, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            qid = p[0]
            s1_order.append(qid)
            cids = [x.strip() for x in p[1].split(",") if x.strip()] if len(p) > 1 and p[1] else []
            cand_dict[qid] = cids
            for cid in cids:
                needed_tgts.add(cid)
                
    print(f"  Loaded {len(s1_order):,} queries referencing {len(needed_tgts):,} unique target candidates.")
    
    print(f"[2/4] Loading raw target database into compact memory ({len(needed_tgts):,} records)...")
    tgt_raw = {}
    for fn in ["test_source2.tsv", "test_source3.tsv"]:
        path = BASE_DIR / fn
        if not path.exists(): continue
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                eid = p[0]
                if eid in needed_tgts:
                    name = p[1]
                    addr = p[2] if len(p) > 2 else ""
                    ctry = p[3].strip() if len(p) > 3 else ""
                    tgt_raw[eid] = (name, addr, ctry)
                    
    print(f"  Target raw database ready: {len(tgt_raw):,} records ({time.time()-t0:.1f}s).")
    
    print("[3/4] Loading Source 1 test queries...")
    s1_norm = {}
    with open(BASE_DIR / "test_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            qid = p[0]
            name = p[1]
            addr = p[2] if len(p) > 2 else ""
            ctry = p[3].strip() if len(p) > 3 else ""
            s1_norm[qid] = (normalize_name(name), normalize_address(addr, ctry))
            
    print(f"  Source 1 normalized: {len(s1_norm):,} queries.")
    
    print(f"[4/4] Streaming batch inference across {len(s1_order):,} queries...")
    match_dict = {}
    CHUNK_SIZE = 50000
    
    n_total = len(s1_order)
    t_start_scoring = time.time()
    
    # Process queries in chunks of 50,000 to keep memory footprint flat (< 1.5 GB)
    for chunk_start in range(0, n_total, CHUNK_SIZE):
        chunk_end = min(n_total, chunk_start + CHUNK_SIZE)
        chunk_qids = s1_order[chunk_start:chunk_end]
        
        # 1. Identify all targets needed for this chunk and normalize them
        chunk_tgts_needed = set()
        for qid in chunk_qids:
            for cid in cand_dict[qid]:
                if cid in tgt_raw:
                    chunk_tgts_needed.add(cid)
                    
        chunk_tgt_norm = {}
        for cid in chunk_tgts_needed:
            name, addr, ctry = tgt_raw[cid]
            chunk_tgt_norm[cid] = (normalize_name(name), normalize_address(addr, ctry))
            
        # 2. Extract features
        batch_feats = []
        batch_counts = []
        for qid in chunk_qids:
            qn, qa = s1_norm[qid]
            cids = cand_dict[qid]
            cnt = 0
            for cid in cids:
                if cid in chunk_tgt_norm:
                    tn, ta = chunk_tgt_norm[cid]
                    batch_feats.append(compute_pair_features(qn, qa, tn, ta))
                    cnt += 1
            batch_counts.append(cnt)
            
        # 3. Vectorized LightGBM prediction
        if batch_feats:
            X = np.array(batch_feats, dtype=np.float32)
            probs = model.predict_proba(X)[:, 1]
            
            offset = 0
            for i, qid in enumerate(chunk_qids):
                cnt = batch_counts[i]
                cids = cand_dict[qid]
                if cnt == 0:
                    match_dict[qid] = ""
                else:
                    q_probs = probs[offset:offset+cnt]
                    sorted_idx = np.argsort(q_probs)[::-1]
                    
                    chosen = []
                    top_idx = sorted_idx[0]
                    if q_probs[top_idx] >= base_th:
                        chosen.append(cids[top_idx])
                        for s_idx in sorted_idx[1:]:
                            if q_probs[s_idx] >= sib_th:
                                chosen.append(cids[s_idx])
                            else:
                                break
                    match_dict[qid] = ",".join(chosen)
                    offset += cnt
        else:
            for qid in chunk_qids:
                match_dict[qid] = ""
                
        # Free chunk memory
        del chunk_tgt_norm
        del batch_feats
        
        done = chunk_end
        elapsed = time.time() - t_start_scoring
        rate = done / max(1.0, elapsed)
        pct = (done / n_total) * 100.0
        print(f"  Progress: {done:,} / {n_total:,} queries ({pct:.1f}%) | Speed: {rate:.0f} q/s | Elapsed: {elapsed:.1f}s")
        
    print(f"All {len(s1_order):,} queries scored successfully!")
    
    # Write output files
    print(f"Writing {OUT_MATCH_FILE} and {FINAL_MATCH_FILE}...")
    for out_p in [OUT_MATCH_FILE, FINAL_MATCH_FILE]:
        with open(out_p, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for qid in s1_order:
                f.write(f"{qid}\t{match_dict.get(qid, '')}\n")
                
    n_singletons = sum(1 for m in match_dict.values() if not m)
    n_matched = len(match_dict) - n_singletons
    print(f"Summary: {len(match_dict):,} total queries | {n_matched:,} matched ({n_matched/len(match_dict)*100:.2f}%) | {n_singletons:,} singletons ({n_singletons/len(match_dict)*100:.2f}%)")
    print(f"Total Pipeline Runtime: {time.time()-t0:.1f}s!")

if __name__ == "__main__":
    main()
