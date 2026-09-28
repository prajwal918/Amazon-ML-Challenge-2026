#!/usr/bin/env python3
"""
Self-contained Fast Grid Tuner for Grandmaster Decision Rules.
Optimizes Macro F0.5 across thresholds to achieve >= 0.90+.
"""

import sys, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from rapidfuzz import fuzz
from anyascii import anyascii

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

TRAIN_DIR = Path("student_resource/dataset/train")

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

DOMAINS = (".com", ".net", ".org", ".co.in", ".in", ".fr", ".co", ".io", ".info", ".biz", ".org.in", ".gov.in")
LEGAL_SUFFIXES = [
    r"\bpvt\.?\s*ltd\.?\b", r"\bprivate\s+limited\b", r"\bltd\.?\b", r"\bllc\b",
    r"\bl\.l\.c\.?\b", r"\binc\.?\b", r"\bcorp\.?\b", r"\bcorporation\b", r"\bco\.?\b",
    r"\bcompany\b", r"\bgmbh\b", r"\bsarl\b", r"\bs\.a\.r\.l\.?\b", r"\bsas\b",
    r"\bs\.a\.s\.?\b", r"\beurl\b", r"\bplc\b", r"\bllp\b", r"\bsa\b", r"\bcenter\b",
    r"\benterprises\b", r"\bservices\b", r"\bholdings\b", r"\bgroup\b", r"\bgroupe\b",
    r"\bindustries\b", r"\bassociates\b", r"\bconsulting\b", r"\bsolutions\b",
    r"\binternational\b", r"\bintl\b"
]
LEGAL_SUFFIX_RE = re.compile("|".join(LEGAL_SUFFIXES), re.IGNORECASE)
STOPWORDS = {"the", "and", "of", "a", "an", "in", "to", "for", "at", "on", "by", "null", "near", "opp", "no", "mr", "ms", "sri", "shri", "smt"}

STATE_CANONICAL = {
    "tamil nadu": "TN", "tamilnadu": "TN", "tn": "TN", "karnataka": "KA", "ka": "KA",
    "kerala": "KL", "kl": "KL", "maharashtra": "MH", "mh": "MH", "andhra pradesh": "AP",
    "andhra": "AP", "ap": "AP", "telangana": "TG", "ts": "TG", "tg": "TG", "gujarat": "GJ",
    "gj": "GJ", "rajasthan": "RJ", "rj": "RJ", "uttar pradesh": "UP", "up": "UP",
    "west bengal": "WB", "wb": "WB", "delhi": "DL", "dl": "DL", "punjab": "PB", "pb": "PB",
    "haryana": "HR", "hr": "HR", "bihar": "BR", "br": "BR", "odisha": "OD", "orissa": "OD",
    "madhya pradesh": "MP", "mp": "MP", "assam": "AS", "jharkhand": "JH", "chhattisgarh": "CG",
    "uttarakhand": "UK", "himachal pradesh": "HP", "goa": "GA",
}
US_STATES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia",
    "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt",
    "va", "wa", "wv", "wi", "wy"
}

def extract_state(addr, ctry):
    if not addr: return None
    addr_l = addr.lower()
    if ctry == "India":
        for st, code in STATE_CANONICAL.items():
            if re.search(r'\b' + re.escape(st) + r'\b', addr_l):
                return code
    elif ctry == "US":
        words = re.findall(r'\b[a-zA-Z]{2}\b', addr)
        for w in reversed(words):
            if w.lower() in US_STATES:
                return w.upper()
    return None

def parse_address_numbers(addr, ctry):
    if not addr: return frozenset(), None
    postal = None
    if ctry == "India":
        m = re.findall(r'\b[1-9]\d{5}\b', addr)
        if m: postal = m[0]
    elif ctry in ("US", "France"):
        m = re.findall(r'\b\d{5}\b', addr)
        if m: postal = m[0]
    all_nums = re.findall(r'\b\d+\b', addr)
    street_nums = set()
    for n in all_nums:
        if postal and n == postal: continue
        if len(n) >= 7: continue
        street_nums.add(n.lstrip("0") or "0")
    return frozenset(street_nums), postal

def smart_has_conflict(q_nums, q_post, t_nums, t_post):
    if q_post and t_post and q_post != t_post: return True
    if q_nums and t_nums:
        if q_nums & t_nums: return False
        for qn in q_nums:
            for tn in t_nums:
                shorter, longer = (qn, tn) if len(qn) <= len(tn) else (tn, qn)
                if len(shorter) >= 3 and shorter in longer: return False
        return True
    return False

def strip_legal(name):
    return LEGAL_SUFFIX_RE.sub(" ", name)

def compress_name(name):
    n = anyascii(name).lower()
    for ext in DOMAINS:
        if n.endswith(ext): n = n[:-len(ext)]
    n = re.sub(r'[^a-z0-9]', '', n)
    for s in ["pvtltd", "privatelimited", "ltd", "llc", "inc", "corp", "corporation", "co", "company", "sarl", "sas", "enterprises", "solutions", "services"]:
        if n.endswith(s): n = n[:-len(s)]
    return n

def prep_record(name, addr, ctry):
    n_asc = anyascii(name).lower()
    a_asc = anyascii(addr).lower()
    core = strip_legal(n_asc).strip()
    core_clean = re.sub(r'[^a-z0-9\s]', ' ', core)
    comp = compress_name(name)
    nums, post = parse_address_numbers(addr, ctry)
    st = extract_state(addr, ctry)
    tokens = frozenset(t for t in core_clean.split() if t not in STOPWORDS and len(t) > 1)
    addr_nonum = re.sub(r'\d+', ' ', a_asc).strip()
    is_latin = all(ord(c) < 256 for c in name if c.isalpha())
    return {
        "name": n_asc, "addr": a_asc, "core": core, "comp": comp,
        "nums": nums, "postal": post, "state": st, "tokens": tokens,
        "addr_nonum": addr_nonum, "ctry": ctry, "is_latin": is_latin
    }

def main():
    print("Loading 5,000 GT queries...")
    gt = {}
    with open(TRAIN_DIR / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i >= 5000: break
            p = line.rstrip("\r\n").split("\t")
            gt[p[0]] = set(x.strip() for x in p[1].split(",") if x.strip()) if len(p) > 1 and p[1] else set()
            
    needed_s1 = set(gt.keys())
    needed_tgts = {tid for tgts in gt.values() for tid in tgts}
    
    s1_entities = {}
    with open(TRAIN_DIR / "train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in needed_s1:
                s1_entities[p[0]] = (p[1], p[2] if len(p) > 2 else "", p[3].strip() if len(p) > 3 else "")
                
    target_entities = {}
    distractors_loaded = 0
    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        with open(TRAIN_DIR / fn, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                eid = p[0]
                if eid in needed_tgts:
                    target_entities[eid] = (p[1], p[2] if len(p) > 2 else "", p[3].strip() if len(p) > 3 else "")
                elif distractors_loaded < 50000:
                    target_entities[eid] = (p[1], p[2] if len(p) > 2 else "", p[3].strip() if len(p) > 3 else "")
                    distractors_loaded += 1
                    
    s1_prep = {sid: prep_record(n, a, c) for sid, (n, a, c) in s1_entities.items()}
    tgt_prep = {tid: prep_record(n, a, c) for tid, (n, a, c) in target_entities.items()}
    
    comp_idx = defaultdict(list)
    token_idx = defaultdict(list)
    num_addr_idx = defaultdict(list)
    
    for tid, t in tgt_prep.items():
        if t["comp"] and len(t["comp"]) >= 4:
            comp_idx[(t["ctry"], t["comp"])].append(tid)
        for tok in t["tokens"]:
            if len(tok) >= 3:
                token_idx[(t["ctry"], tok)].append(tid)
        for num in t["nums"]:
            if len(num) >= 2:
                num_addr_idx[(t["ctry"], num)].append(tid)
                
    print("Pre-evaluating candidate feature vectors...")
    query_candidates = {}
    for sid, q in s1_prep.items():
        q_ctry = q["ctry"]
        cands = set()
        if q["comp"] and (q_ctry, q["comp"]) in comp_idx:
            cands.update(comp_idx[(q_ctry, q["comp"])])
        for tok in q["tokens"]:
            key = (q_ctry, tok)
            if key in token_idx and len(token_idx[key]) <= 400:
                cands.update(token_idx[key])
        for num in q["nums"]:
            key = (q_ctry, num)
            if key in num_addr_idx and len(num_addr_idx[key]) <= 250:
                cands.update(num_addr_idx[key])
                
        pairs = []
        for tid in cands:
            t = tgt_prep[tid]
            if t["ctry"] != q_ctry: continue
            if q["state"] and t["state"] and q["state"] != t["state"]: continue
            if smart_has_conflict(q["nums"], q["postal"], t["nums"], t["postal"]): continue
            
            num_agree = bool(q["nums"] and t["nums"] and (q["nums"] & t["nums"]))
            core_sort = fuzz.token_sort_ratio(q["core"], t["core"]) if (q["core"] and t["core"]) else 0
            asim = fuzz.token_sort_ratio(q["addr"], t["addr"]) if (q["addr"] and t["addr"]) else 0
            street_sim = fuzz.token_set_ratio(q["addr_nonum"], t["addr_nonum"]) if (q["addr_nonum"] and t["addr_nonum"]) else 0
            is_exact_comp = bool(q["comp"] and t["comp"] and q["comp"] == t["comp"])
            tok_overlap = len(q["tokens"] & t["tokens"])
            has_no_addr = not t["addr"] or not q["addr"]
            
            pairs.append({
                "tid": tid, "is_exact_comp": is_exact_comp, "num_agree": num_agree,
                "core_sort": core_sort, "asim": asim, "street_sim": street_sim,
                "tok_overlap": tok_overlap, "has_no_addr": has_no_addr,
                "is_latin": t["is_latin"], "q_ctry": q_ctry
            })
        query_candidates[sid] = pairs
        
    print("Running Fast Threshold Grid Search...")
    best_score = 0.0
    best_params = None
    
    for street_thresh in [45, 50, 55, 60, 65]:
        for core_thresh in [25, 30, 35, 40]:
            for high_core in [75, 80, 85]:
                scores = []
                tp = fp = fn = 0
                
                for sid, true_m in gt.items():
                    matches = []
                    for p in query_candidates[sid]:
                        is_match = False
                        if p["is_exact_comp"]:
                            if p["has_no_addr"] or p["street_sim"] >= 15 or p["num_agree"] or p["core_sort"] >= 45:
                                is_match = True
                        elif not p["is_latin"] and p["q_ctry"] == "India":
                            if p["num_agree"] and (p["street_sim"] >= 50 or p["asim"] >= 45):
                                is_match = True
                        elif p["num_agree"] and p["street_sim"] >= street_thresh and (p["core_sort"] >= core_thresh or p["tok_overlap"] >= 1):
                            is_match = True
                        elif p["core_sort"] >= high_core and (p["num_agree"] or p["street_sim"] >= 25 or p["has_no_addr"]):
                            is_match = True
                        elif p["core_sort"] >= 65 and (p["num_agree"] or p["street_sim"] >= 65):
                            is_match = True
                            
                        if is_match: matches.append(p["tid"])
                        
                    pred_set = set(matches[:4])
                    cur_tp = len(pred_set & true_m)
                    cur_fp = len(pred_set - true_m)
                    cur_fn = len(true_m - pred_set)
                    tp += cur_tp; fp += cur_fp; fn += cur_fn
                    scores.append(compute_entity_f_beta(pred_set, true_m, beta=0.5))
                    
                macro_f = np.mean(scores)
                prec = tp / max(1, tp + fp)
                rec = tp / max(1, tp + fn)
                
                if macro_f > best_score:
                    best_score = macro_f
                    best_params = (street_thresh, core_thresh, high_core)
                    print(f"--> BEST: F0.5 = {macro_f:.4f} | Prec = {prec:.4f}, Rec = {rec:.4f}, TP = {tp:,}, FP = {fp:,} | street={street_thresh}, core={core_thresh}, high_core={high_core}")

    print("\n" + "=" * 70)
    print(f"GRID SEARCH COMPLETE! Peak Macro F0.5: {best_score:.4f} with params {best_params}")
    print("=" * 70)

if __name__ == "__main__":
    main()
