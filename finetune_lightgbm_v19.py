#!/usr/bin/env python3
"""
================================================================================
Amazon ML Challenge 2026 -- Finetune LightGBM Resolver V19
================================================================================
Trains a 22-feature ultra-calibrated LightGBM model on large-scale realistic
candidate pairs (ground truth + blocking hard negatives + distractors).
Evaluates strictly on disjoint held-out validation queries.
================================================================================
"""

import sys
import os
import re
import time
import math
import random
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

# ==============================================================================
# NORMALIZATION
# ==============================================================================

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
    
    # 3-grams
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
    # Disambiguate "fl" for floor
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

# ==============================================================================
# 22 PAIRWISE FEATURES
# ==============================================================================

FEATURE_NAMES = [
    "name_token_sort", "name_token_set", "name_partial",
    "name_jw", "name_core_jw", "name_core_sort", "name_core_set",
    "name_exact_core", "name_exact_comp", "name_len_ratio", "name_len_diff",
    "name_token_jaccard", "name_tok_overlap_cnt", "name_qgram3_jaccard", "used_alt_name",
    "addr_house_match", "addr_house_conflict",
    "addr_state_match", "addr_state_conflict",
    "addr_postal_match", "addr_postal_conflict",
    "addr_token_sort", "addr_token_set", "addr_token_jaccard",
    "addr_target_empty", "name_x_addr", "core_x_addr_set"
]

def _name_variants(norm):
    variants = []
    seen = set()
    for field in ("full", "core", "leet", "dba_alt"):
        v = norm.get(field) or ""
        if v and v not in seen:
            variants.append((field, v))
            seen.add(v)
    return variants or [("full", "")]

def compute_pair_features(q_norm, q_addr, t_norm, t_addr):
    # --- Name similarities ---
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
    
    # Token Jaccard
    inter_tok = len(q_norm["tokens"] & t_norm["tokens"])
    union_tok = len(q_norm["tokens"] | t_norm["tokens"])
    tok_jaccard = inter_tok / union_tok if union_tok > 0 else 0.0
    
    # 3-gram Jaccard
    inter_qg = len(q_norm["qgrams"] & t_norm["qgrams"])
    union_qg = len(q_norm["qgrams"] | t_norm["qgrams"])
    qgram_jaccard = inter_qg / union_qg if union_qg > 0 else 0.0
    
    used_alt = 1.0 if (q_norm["dba_alt"] or t_norm["dba_alt"]) else 0.0
    
    # --- Address similarities & conflicts ---
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

# ==============================================================================
# OFFICIAL EVALUATION METRIC (MACRO F0.5)
# ==============================================================================

def compute_entity_f_beta(pred_set, true_set, beta=0.5):
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    if len(pred_set) == 0 or tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta_sq = beta ** 2
    return ((1 + beta_sq) * precision * recall) / ((beta_sq * precision) + recall)

# ==============================================================================
# TRAINING & FINE-TUNING PIPELINE
# ==============================================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("FINETUNING LIGHTGBM RESOLVER V19 ON MULTI-SOURCE BENCHMARK")
    print("=" * 80)
    
    TRAIN_DIR = Path("student_resource/dataset/train")
    
    # 1. Load Ground Truth (sample 20,000 queries)
    print("[1/6] Loading Ground Truth labels...")
    all_gt = {}
    with open(TRAIN_DIR / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i >= 20000: break
            p = line.rstrip("\r\n").split("\t")
            all_gt[p[0]] = set(x.strip() for x in p[1].split(",") if x.strip()) if len(p) > 1 and p[1] else set()
            
    print(f"  Loaded {len(all_gt):,} ground truth queries.")
    needed_s1 = set(all_gt.keys())
    needed_tgts = {tid for tgts in all_gt.values() for tid in tgts}
    print(f"  Total true matching target entities: {len(needed_tgts):,}")
    
    # 2. Load Source 1 entities
    print("[2/6] Loading Source 1 records...")
    s1_records = {}
    with open(TRAIN_DIR / "train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in needed_s1:
                s1_records[p[0]] = (p[1], p[2] if len(p) > 2 else "", p[3].strip() if len(p) > 3 else "")
                if len(s1_records) >= len(needed_s1):
                    break
    print(f"  Loaded {len(s1_records):,} Source 1 query records.")

    # 3. Load Target entities: all true matches + 100,000 negative distractors
    print("[3/6] Loading Target records (true matches + 100,000 hard distractors)...")
    target_records = {}
    distractors_loaded = 0
    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        with open(TRAIN_DIR / fn, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                eid = p[0]
                if eid in needed_tgts:
                    target_records[eid] = (p[1], p[2] if len(p) > 2 else "", p[3].strip() if len(p) > 3 else "")
                elif distractors_loaded < 100000:
                    target_records[eid] = (p[1], p[2] if len(p) > 2 else "", p[3].strip() if len(p) > 3 else "")
                    distractors_loaded += 1
                    
    print(f"  Loaded {len(target_records):,} target records ({len(needed_tgts):,} true, {distractors_loaded:,} distractors).")
    
    # 4. Precompute normalizations
    print("[4/6] Precomputing normalizations...")
    s1_norm = {sid: (normalize_name(n), normalize_address(a, c), c) for sid, (n, a, c) in s1_records.items()}
    tgt_norm = {tid: (normalize_name(n), normalize_address(a, c), c) for tid, (n, a, c) in target_records.items()}
    
    # Build inverted index for hard negative generation
    comp_idx = defaultdict(list)
    token_idx = defaultdict(list)
    hn_idx = defaultdict(list)
    
    for tid, (tn, ta, ctry) in tgt_norm.items():
        if tn["comp"] and len(tn["comp"]) >= 3:
            comp_idx[(ctry, tn["comp"][:6])].append(tid)
        for tok in tn["tokens"]:
            if len(tok) >= 3:
                token_idx[(ctry, tok)].append(tid)
        if ta["house_number"]:
            hn_idx[(ctry, ta["house_number"])].append(tid)

    # 5. Generate Candidate Pairs (True Positives + Hard Negatives)
    print("[5/6] Generating candidate pairs and extracting 27 features...")
    qids = list(s1_records.keys())
    random.seed(42)
    random.shuffle(qids)
    
    val_split = int(len(qids) * 0.20)
    val_qids = set(qids[:val_split])
    train_qids = set(qids[val_split:])
    print(f"  Split: {len(train_qids):,} Train queries, {len(val_qids):,} Validation queries (Disjoint).")
    
    train_X, train_y, train_q = [], [], []
    val_candidates = defaultdict(list) # qid -> list of (tid, features, label)
    
    n_pos = n_neg = 0
    
    for qid in qids:
        qn, qa, q_ctry = s1_norm[qid]
        true_set = all_gt[qid]
        is_val = qid in val_qids
        
        # Candidate generation for this query
        cands = set(true_set) # Always include ground truth
        
        # Add hard negatives via blocking
        if qn["comp"] and len(qn["comp"]) >= 3:
            hits = comp_idx.get((q_ctry, qn["comp"][:6]), [])
            cands.update(hits[:5])
        for tok in qn["tokens"]:
            hits = token_idx.get((q_ctry, tok), [])
            cands.update(hits[:3])
            if len(cands) >= 20: break
        if qa["house_number"]:
            hits = hn_idx.get((q_ctry, qa["house_number"]), [])
            cands.update(hits[:3])
            
        # Filter to targets that exist in our target pool
        cands = [tid for tid in cands if tid in tgt_norm]
        
        # Extract features
        for tid in cands:
            tn, ta, _ = tgt_norm[tid]
            feats = compute_pair_features(qn, qa, tn, ta)
            label = 1 if tid in true_set else 0
            
            if is_val:
                val_candidates[qid].append((tid, feats, label))
            else:
                train_X.append(feats)
                train_y.append(label)
                train_q.append(qid)
                if label == 1: n_pos += 1
                else: n_neg += 1
                
    print(f"  Training Set: {len(train_X):,} pairs ({n_pos:,} positive, {n_neg:,} negative, ratio: 1:{n_neg/max(1, n_pos):.1f})")
    
    # 6. Train LightGBM Model
    print("[6/6] Training LightGBM with Early Stopping...")
    X_train = np.array(train_X, dtype=np.float32)
    y_train = np.array(train_y, dtype=np.int32)
    
    # Prepare validation matrix
    val_X_flat, val_y_flat = [], []
    for qid in val_qids:
        for tid, feats, label in val_candidates[qid]:
            val_X_flat.append(feats)
            val_y_flat.append(label)
            
    X_val = np.array(val_X_flat, dtype=np.float32)
    y_val = np.array(val_y_flat, dtype=np.int32)
    print(f"  Validation Set: {len(X_val):,} pairs ({sum(y_val):,} positive, {len(y_val)-sum(y_val):,} negative)")
    
    clf = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=7,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary",
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=True)]
    )
    
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE RANKING (Top 15):")
    print("=" * 80)
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for r, idx in enumerate(sorted_idx[:15], 1):
        print(f"  {r:2d}. {FEATURE_NAMES[idx]:<25} : {importances[idx]:6d}")
        
    # Predict validation probabilities
    val_probs = clf.predict_proba(X_val)[:, 1]
    
    # Re-assemble validation predictions per query
    offset = 0
    val_preds_by_q = {}
    for qid in val_qids:
        pairs = val_candidates[qid]
        cnt = len(pairs)
        if cnt == 0:
            val_preds_by_q[qid] = []
        else:
            q_probs = val_probs[offset:offset+cnt]
            val_preds_by_q[qid] = [(pairs[i][0], q_probs[i]) for i in range(cnt)]
            offset += cnt
            
    # Decision rule grid search for optimal Macro F0.5
    print("\n" + "=" * 80)
    print("GRID SEARCH DECISION RULES ON HELD-OUT 4,000 VALIDATION QUERIES")
    print("=" * 80)
    
    best_macro_f = 0.0
    best_config = None
    
    for base_th in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        for sib_th in [0.35, 0.40, 0.45, 0.50]:
            if sib_th > base_th: continue
            
            scores = []
            tp = fp = fn = 0
            singletons_correct = 0
            n_singletons = 0
            
            for qid in val_qids:
                true_set = all_gt[qid]
                if len(true_set) == 0: n_singletons += 1
                
                pairs = val_preds_by_q[qid]
                if not pairs:
                    pred_set = set()
                else:
                    sorted_p = sorted(pairs, key=lambda x: x[1], reverse=True)
                    pred_set = set()
                    
                    # Top candidate exceeds base threshold
                    if sorted_p[0][1] >= base_th:
                        pred_set.add(sorted_p[0][0])
                        # Sibling corroboration
                        for tid, p in sorted_p[1:]:
                            if p >= sib_th:
                                pred_set.add(tid)
                            else:
                                break
                                
                cur_tp = len(pred_set & true_set)
                cur_fp = len(pred_set - true_set)
                cur_fn = len(true_set - pred_set)
                tp += cur_tp; fp += cur_fp; fn += cur_fn
                if len(true_set) == 0 and len(pred_set) == 0:
                    singletons_correct += 1
                    
                scores.append(compute_entity_f_beta(pred_set, true_set, beta=0.5))
                
            macro_f = np.mean(scores)
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            
            if macro_f > best_macro_f:
                best_macro_f = macro_f
                best_config = (base_th, sib_th, prec, rec, singletons_correct, n_singletons)
                print(f"  * New Best: BaseTh={base_th:.2f}, SibTh={sib_th:.2f} -> Macro F0.5 = {macro_f:.4f} | Prec = {prec*100:.2f}% | Rec = {rec*100:.2f}%")
            else:
                print(f"    Tested:   BaseTh={base_th:.2f}, SibTh={sib_th:.2f} -> Macro F0.5 = {macro_f:.4f} | Prec = {prec*100:.2f}% | Rec = {rec*100:.2f}%")
                
    base_th, sib_th, prec, rec, s_cor, s_tot = best_config
    print("\n" + "=" * 80)
    print("OPTIMAL VALIDATION PERFORMANCE (DISJOINT HELD-OUT QUERIES):")
    print("=" * 80)
    print(f"Macro F0.5 Score:        {best_macro_f:.4f}")
    print(f"Precision:               {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall:                  {rec:.4f} ({rec*100:.2f}%)")
    print(f"Singleton Accuracy:      {s_cor}/{s_tot} ({s_cor/max(1, s_tot)*100:.2f}%)")
    print(f"Optimal Base Threshold:  {base_th:.2f}")
    print(f"Optimal Sibling Thresh:  {sib_th:.2f}")
    print(f"Total Elapsed Time:      {time.time() - t0:.1f}s")
    print("=" * 80)
    
    # Save the trained model bundle
    out_model_path = "model_finetuned_v19.pkl"
    joblib.dump({
        "model": clf,
        "feature_names": FEATURE_NAMES,
        "base_threshold": base_th,
        "sibling_threshold": sib_th,
        "macro_f05": best_macro_f
    }, out_model_path)
    print(f"Successfully saved finetuned model bundle to: {out_model_path}")

if __name__ == "__main__":
    main()
