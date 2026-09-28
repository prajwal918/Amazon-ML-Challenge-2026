#!/usr/bin/env python3
import sys, re, unicodedata, time
from collections import defaultdict
from pathlib import Path
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from anyascii import anyascii
from rapidfuzz import fuzz

TRAIN = Path("student_resource/dataset/train")

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

def is_text_latin(s):
    return all(ord(c) < 256 for c in s if c.isalpha())

# Load 1,000 GT queries
gt = {}
with open(TRAIN / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 1000: break
        p = line.rstrip("\r\n").split("\t")
        gt[p[0]] = set(x.strip() for x in p[1].split(",") if x.strip()) if len(p) > 1 and p[1] else set()

needed_s1 = set(gt.keys())
needed_tgts = {tid for tgts in gt.values() for tid in tgts}

s1_entities = {}
with open(TRAIN / "train_source1.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if p[0] in needed_s1:
            s1_entities[p[0]] = (p[1], p[2] if len(p) > 2 else "", p[3].strip() if len(p) > 3 else "")

target_entities = {}
distractors_loaded = 0
for fn in ["train_source2.tsv", "train_source3.tsv"]:
    with open(TRAIN / fn, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            eid = p[0]
            if eid in needed_tgts:
                target_entities[eid] = (p[1], p[2] if len(p) > 2 else "", p[3].strip() if len(p) > 3 else "")
            elif distractors_loaded < 20000:
                target_entities[eid] = (p[1], p[2] if len(p) > 2 else "", p[3].strip() if len(p) > 3 else "")
                distractors_loaded += 1

print(f"Loaded {len(s1_entities)} S1 queries, {len(target_entities)} target records ({len(needed_tgts)} true, {distractors_loaded} distractors)")

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

us_fr_zip = re.compile(r'\b\d{5}\b')
in_pin = re.compile(r'\b\d{6}\b')

def get_postal(addr, ctry):
    if ctry in ("US", "France"): return frozenset(us_fr_zip.findall(addr))
    elif ctry == "India": return frozenset(in_pin.findall(addr))
    return frozenset()

def extract_state(addr, ctry):
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

def get_nums(addr):
    return frozenset(re.findall(r'\b\d+\b', addr))

def prep_record(name, addr, ctry):
    n_asc = anyascii(name).lower()
    a_asc = anyascii(addr).lower()
    core = strip_legal(n_asc).strip()
    core_clean = re.sub(r'[^a-z0-9\s]', ' ', core)
    comp = compress_name(name)
    nums = get_nums(a_asc)
    st = extract_state(addr, ctry)
    post = get_postal(addr, ctry)
    tokens = frozenset(t for t in core_clean.split() if t not in STOPWORDS and len(t) > 1)
    addr_nonum = re.sub(r'\d+', ' ', a_asc).strip()
    raw_latin = is_text_latin(name)
    return {
        "name": n_asc, "addr": a_asc, "core": core, "comp": comp,
        "nums": nums, "state": st, "post": post, "tokens": tokens,
        "addr_nonum": addr_nonum, "ctry": ctry, "raw_latin": raw_latin
    }

print("Preprocessing records...")
s1_prep = {sid: prep_record(n, a, c) for sid, (n, a, c) in s1_entities.items()}
tgt_prep = {tid: prep_record(n, a, c) for tid, (n, a, c) in target_entities.items()}

# Inverted index on targets for fast retrieval
comp_idx = defaultdict(list)
token_idx = defaultdict(list)
num_addr_idx = defaultdict(list)

for tid, t in tgt_prep.items():
    if t["comp"] and len(t["comp"]) >= 4:
        comp_idx[t["comp"]].append(tid)
    for tok in t["tokens"]:
        if len(tok) >= 3:
            token_idx[tok].append(tid)
    for num in t["nums"]:
        if len(num) >= 2:
            num_addr_idx[num].append(tid)

preds_v11 = {}
preds_v14 = {}

for sid, q in s1_prep.items():
    q_ctry = q["ctry"]
    cands = set()
    if q["comp"] and q["comp"] in comp_idx: cands.update(comp_idx[q["comp"]])
    for tok in q["tokens"]:
        if tok in token_idx and len(token_idx[tok]) <= 300: cands.update(token_idx[tok])
    for num in q["nums"]:
        if num in num_addr_idx and len(num_addr_idx[num]) <= 200: cands.update(num_addr_idx[num])
        
    m_v11 = []
    m_v14 = []
    
    for tid in cands:
        t = tgt_prep[tid]
        if t["ctry"] != q_ctry: continue
        
        num_agree = bool(q["nums"] and t["nums"] and (q["nums"] & t["nums"]))
        num_conflict = bool(q["nums"] and t["nums"] and not (q["nums"] & t["nums"]))
        
        core_sort = fuzz.token_sort_ratio(q["core"], t["core"]) if (q["core"] and t["core"]) else 0
        asim = fuzz.token_sort_ratio(q["addr"], t["addr"]) if (q["addr"] and t["addr"]) else 0
        street_sim = fuzz.token_set_ratio(q["addr_nonum"], t["addr_nonum"]) if (q["addr_nonum"] and t["addr_nonum"]) else 0
        
        # V11 logic (conservative building check, token Jaccard)
        q_toks = q["tokens"]
        t_toks = t["tokens"]
        n_j = len(q_toks & t_toks) / max(1, len(q_toks | t_toks))
        
        q_atoks = set(re.findall(r"\b[a-z]{2,}\b", q["addr"]))
        t_atoks = set(re.findall(r"\b[a-z]{2,}\b", t["addr"]))
        a_j = len(q_atoks & t_atoks) / max(1, len(q_atoks | t_atoks))
        
        is_v11 = False
        if a_j >= 0.50 or (num_agree and a_j >= 0.28): is_v11 = True
        elif n_j >= 0.70: is_v11 = True
        elif a_j >= 0.35 and n_j >= 0.35: is_v11 = True
        if is_v11: m_v11.append((max(n_j, a_j), tid))
        
        # V14 clean logic (state-aware, strict number, legal stripped)
        if q["state"] and t["state"] and q["state"] != t["state"]: continue
        if num_conflict: continue
        
        is_v14 = False
        if q["comp"] and t["comp"] and q["comp"] == t["comp"]:
            if not t["addr"] or not q["addr"] or street_sim >= 20 or num_agree: is_v14 = True
        elif not t["raw_latin"] and q_ctry == "India" and num_agree and (street_sim >= 65 or asim >= 60):
            is_v14 = True
        elif num_agree and street_sim >= 70 and (core_sort >= 45 or len(q_toks & t_toks) >= 1):
            is_v14 = True
        elif core_sort >= 85 and (num_agree or street_sim >= 35 or not t["addr"]):
            is_v14 = True
        elif core_sort >= 70 and (num_agree or street_sim >= 75):
            is_v14 = True
        if is_v14: m_v14.append((core_sort, tid))
        
    m_v11.sort(reverse=True)
    m_v14.sort(reverse=True)
    preds_v11[sid] = [tid for _, tid in m_v11[:5]]
    preds_v14[sid] = [tid for _, tid in m_v14[:5]]

def score_preds(preds):
    scs = []
    tp = fp = fn = 0
    for sid, true_m in gt.items():
        p = set(preds[sid])
        cur_tp = len(p & true_m)
        cur_fp = len(p - true_m)
        cur_fn = len(true_m - p)
        tp += cur_tp; fp += cur_fp; fn += cur_fn
        scs.append(compute_entity_f_beta(p, true_m, beta=0.5))
    return np.mean(scs), tp / max(1, tp+fp), tp / max(1, tp+fn), tp, fp, fn

f_11, p_11, r_11, tp11, fp11, fn11 = score_preds(preds_v11)
f_14, p_14, r_14, tp14, fp14, fn14 = score_preds(preds_v14)

# Strategy 1: If intersection non-empty -> intersection. If empty -> pick V14
preds_strat1 = {sid: list(set(preds_v11[sid]) & set(preds_v14[sid])) if (set(preds_v11[sid]) & set(preds_v14[sid])) else preds_v14[sid] for sid in gt}
f1, p1, r1, tp1, fp1, _ = score_preds(preds_strat1)

# Strategy 2: If intersection non-empty -> intersection. If empty -> pick V11
preds_strat2 = {sid: list(set(preds_v11[sid]) & set(preds_v14[sid])) if (set(preds_v11[sid]) & set(preds_v14[sid])) else preds_v11[sid] for sid in gt}
f2, p2, r2, tp2, fp2, _ = score_preds(preds_strat2)

# Strategy 3: Intersection strictly, else empty
preds_strat3 = {sid: list(set(preds_v11[sid]) & set(preds_v14[sid])) for sid in gt}
f3, p3, r3, tp3, fp3, _ = score_preds(preds_strat3)

# Strategy 4: Union (up to 5)
preds_strat4 = {sid: list(set(preds_v11[sid]) | set(preds_v14[sid]))[:5] for sid in gt}
f4, p4, r4, tp4, fp4, _ = score_preds(preds_strat4)

# Strategy 5: V11 + V14 high-confidence (exact compressed + Indic non-latin)
preds_strat5 = {}
for sid in gt:
    res = list(preds_v11[sid])
    q = s1_prep[sid]
    for tid in preds_v14[sid]:
        t = tgt_prep[tid]
        # Only add if exact compressed match or non-latin
        if (q["comp"] and t["comp"] and q["comp"] == t["comp"]) or (not t["raw_latin"] and q["ctry"] == "India"):
            if tid not in res and len(res) < 5:
                res.append(tid)
    preds_strat5[sid] = res

f5, p5, r5, tp5, fp5, _ = score_preds(preds_strat5)

print("=" * 70)
print(f"V11 Baseline:                 F0.5 = {f_11:.4f}, Prec = {p_11:.4f}, Rec = {r_11:.4f}, TP = {tp11}, FP = {fp11}")
print(f"V14 Clean:                    F0.5 = {f_14:.4f}, Prec = {p_14:.4f}, Rec = {r_14:.4f}, TP = {tp14}, FP = {fp14}")
print(f"Strat 1 (Inter else V14):     F0.5 = {f1:.4f}, Prec = {p1:.4f}, Rec = {r1:.4f}, TP = {tp1}, FP = {fp1}")
print(f"Strat 2 (Inter else V11):     F0.5 = {f2:.4f}, Prec = {p2:.4f}, Rec = {r2:.4f}, TP = {tp2}, FP = {fp2}")
print(f"Strat 3 (Strict Inter):       F0.5 = {f3:.4f}, Prec = {p3:.4f}, Rec = {r3:.4f}, TP = {tp3}, FP = {fp3}")
print(f"Strat 4 (Full Union):         F0.5 = {f4:.4f}, Prec = {p4:.4f}, Rec = {r4:.4f}, TP = {tp4}, FP = {fp4}")
print(f"Strat 5 (V11 + V14 High-Conf):F0.5 = {f5:.4f}, Prec = {p5:.4f}, Rec = {r5:.4f}, TP = {tp5}, FP = {fp5}")
print("=" * 70)
