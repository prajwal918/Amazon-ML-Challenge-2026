import sys, re
import pandas as pd
from collections import defaultdict
from pathlib import Path
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from anyascii import anyascii
from rapidfuzz import fuzz

DATA_DIR = Path('student_resource/dataset/train')
gt = pd.read_csv(DATA_DIR / 'train_ground_truth.tsv', sep='\t', nrows=100)
gt_dict = {}
all_tgts = set()
for _, r in gt.iterrows():
    qid = r['source1_entity_id']
    if pd.isna(r['matched_entity_ids']):
        gt_dict[qid] = []
    else:
        tgts = [x.strip() for x in str(r['matched_entity_ids']).split(',') if x.strip()]
        gt_dict[qid] = tgts
        all_tgts.update(tgts)

s1_map = {}
with open(DATA_DIR / 'train_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in gt_dict:
            s1_map[p[0]] = (p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else '')

target_records = {}
distractors_loaded = 0
for fn in ['train_source2.tsv', 'train_source3.tsv']:
    with open(DATA_DIR / fn, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            eid = p[0]
            if eid in all_tgts:
                target_records[eid] = (p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else '')
            elif distractors_loaded < 5000:
                target_records[eid] = (p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else '')
                distractors_loaded += 1

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

# Indian states & UTs
INDIAN_STATES = {
    "tamil nadu", "tamilnadu", "tn", "karnataka", "ka", "kerala", "kl", "maharashtra", "mh",
    "andhra pradesh", "andhra", "ap", "telangana", "ts", "tg", "gujarat", "gj", "rajasthan", "rj",
    "uttar pradesh", "up", "west bengal", "wb", "delhi", "dl", "punjab", "pb", "haryana", "hr",
    "bihar", "br", "odisha", "orissa", "or", "madhya pradesh", "mp", "assam", "as", "jharkhand", "jh",
    "chhattisgarh", "cg", "uttarakhand", "uk", "himachal pradesh", "hp", "goa", "ga"
}

STATE_CANONICAL = {
    "tamil nadu": "TN", "tamilnadu": "TN", "tn": "TN",
    "karnataka": "KA", "ka": "KA",
    "kerala": "KL", "kl": "KL",
    "maharashtra": "MH", "mh": "MH",
    "andhra pradesh": "AP", "andhra": "AP", "ap": "AP",
    "telangana": "TG", "ts": "TG", "tg": "TG",
    "gujarat": "GJ", "gj": "GJ",
    "rajasthan": "RJ", "rj": "RJ",
    "uttar pradesh": "UP", "up": "UP",
    "west bengal": "WB", "wb": "WB",
    "delhi": "DL", "dl": "DL",
    "punjab": "PB", "pb": "PB",
    "haryana": "HR", "hr": "HR",
    "bihar": "BR", "br": "BR",
    "odisha": "OD", "orissa": "OD", "or": "OD",
    "madhya pradesh": "MP", "mp": "MP",
    "assam": "AS", "as": "AS",
}

US_STATES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia",
    "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt",
    "va", "wa", "wv", "wi", "wy"
}

def extract_state(addr, ctry):
    addr_l = addr.lower()
    if ctry == "India":
        for st, code in STATE_CANONICAL.items():
            if re.search(r'\b' + re.escape(st) + r'\b', addr_l):
                return code
    elif ctry == "US":
        words = re.findall(r'\b[a-zA-Z]{2}\b', addr)
        for w in reversed(words):
            wl = w.lower()
            if wl in US_STATES:
                return wl.upper()
    return None

def clean_text(s):
    s = anyascii(str(s)).lower().strip()
    return " ".join(s.split())

def strip_legal(s):
    s = LEGAL_SUFFIX_RE.sub(" ", s)
    return " ".join(s.split())

def compress_name(n):
    n = clean_text(n)
    for ext in DOMAINS:
        if n.endswith(ext): n = n[:-len(ext)].strip()
    n = re.sub(r'[^a-z0-9]', '', n)
    for suf in ("corporation", "corporate", "private", "limited", "holdings", "services", 
                "enterprises", "company", "corp", "pvt", "ltd", "llc", "inc", "gmbh", 
                "sarl", "sas", "sasu", "eurl", "dba", "group", "groupe", "center"):
        if n.endswith(suf): n = n[:-len(suf)]
    return n

def get_tokens(s):
    toks = re.findall(r'[a-zA-Z0-9]+', clean_text(s))
    return frozenset(t for t in toks if len(t) > 2 and t not in STOPWORDS)

def get_nums(s):
    raw = re.findall(r'\b\d+\b', s)
    return frozenset(n.lstrip("0") or "0" for n in raw if len(n) >= 1)

def match_pair(qname, qaddr, qctry, tname, taddr, tctry):
    if qctry != tctry:
        return False, 0.0

    qn_clean = clean_text(qname)
    tn_clean = clean_text(tname)
    qa_clean = clean_text(qaddr)
    ta_clean = clean_text(taddr)

    # State conflict check (Never merge across different states!)
    q_st = extract_state(qaddr, qctry)
    t_st = extract_state(taddr, tctry)
    if q_st and t_st and q_st != t_st:
        return False, 0.0

    qcomp = compress_name(qname)
    tcomp = compress_name(tname)

    qn_core = strip_legal(qn_clean)
    tn_core = strip_legal(tn_clean)
    q_core_tokens = get_tokens(qn_core)
    t_core_tokens = get_tokens(tn_core)
    shared_core_tokens = q_core_tokens & t_core_tokens

    qnums = get_nums(qaddr)
    tnums = get_nums(taddr)

    num_agree = bool(qnums and tnums and (qnums & tnums))
    num_conflict = bool(qnums and tnums and not (qnums & tnums))

    core_sort = fuzz.token_sort_ratio(qn_core, tn_core) if (qn_core and tn_core) else 0
    core_set = fuzz.token_set_ratio(qn_core, tn_core) if (qn_core and tn_core) else 0
    core_nsim = max(core_sort, core_set)

    asim_sort = fuzz.token_sort_ratio(qa_clean, ta_clean) if (qa_clean and ta_clean) else 0
    asim_set = fuzz.token_set_ratio(qa_clean, ta_clean) if (qa_clean and ta_clean) else 0
    asim = max(asim_sort, asim_set)

    qa_nonum = re.sub(r'\d+', ' ', qa_clean).strip()
    ta_nonum = re.sub(r'\d+', ' ', ta_clean).strip()
    street_sim = fuzz.token_set_ratio(qa_nonum, ta_nonum) if (qa_nonum and ta_nonum) else 0

    if num_conflict:
        is_num_typo = False
        if core_nsim >= 65 and street_sim >= 75:
            for qn in qnums:
                for tn in tnums:
                    if len(qn) >= 2 and len(tn) >= 2:
                        if qn in tn or tn in qn or abs(len(qn) - len(tn)) <= 1 or fuzz.ratio(qn, tn) >= 66:
                            is_num_typo = True
                            break
                if is_num_typo: break
        if not is_num_typo:
            return False, 0.0

    # 1. Exact compressed name match (domain, full name, punctuation)
    if qcomp and tcomp and qcomp == tcomp:
        if not ta_clean or not qa_clean or street_sim >= 20 or num_agree:
            return True, 100.0

    # 2. Corrupted / masked / Indic name, but address matches with building number
    if (num_agree or street_sim >= 85) and asim >= 60 and street_sim >= 65:
        if core_nsim >= 30 or not tn_core or shared_core_tokens or len(shared_core_tokens) >= 1 or len(t_core_tokens) <= 1:
            return True, 90.0

    # 3. High address similarity (when address is long and identical)
    if asim >= 75 and len(qa_clean) > 15 and (core_nsim >= 35 or not tn_core or shared_core_tokens):
        return True, 85.0

    # 4. High core name similarity with address corroboration OR empty target address
    if core_nsim >= 80:
        if not ta_clean or not qa_clean:
            if len(shared_core_tokens) >= 1 or len(qn_core) >= 8:
                return True, 80.0
        elif asim >= 25 or num_agree or street_sim >= 40:
            return True, 80.0

    # 5. Moderate name (shared at least 2 tokens) when target address is empty
    if not ta_clean and len(shared_core_tokens) >= 2 and core_nsim >= 65:
        return True, 78.0

    # 6. Moderate name + Strong address
    if core_nsim >= 50 and (asim >= 50 or num_agree or street_sim >= 65):
        return True, 75.0

    return False, 0.0

pred_dict = defaultdict(list)
for qid in gt_dict:
    qname, qaddr, qctry = s1_map.get(qid, ('', '', ''))
    for tid, (tname, taddr, tctry) in target_records.items():
        is_m, score = match_pair(qname, qaddr, qctry, tname, taddr, tctry)
        if is_m:
            pred_dict[qid].append((score, tid))

final_preds = {}
for qid in gt_dict:
    matches = sorted(pred_dict[qid], key=lambda x: x[0], reverse=True)
    final_preds[qid] = [tid for sc, tid in matches[:4]]

b2 = 0.25
scores = []
tp_tot = fp_tot = fn_tot = 0
for qid in gt_dict:
    preds = set(final_preds[qid])
    gts = set(gt_dict[qid])
    cur_tp = len(preds & gts)
    cur_fp = len(preds - gts)
    cur_fn = len(gts - preds)
    tp_tot += cur_tp
    fp_tot += cur_fp
    fn_tot += cur_fn
    if len(gts) == 0:
        scores.append(1.0 if len(preds) == 0 else 0.0)
    else:
        denom = 1.25 * cur_tp + 0.25 * cur_fn + cur_fp
        scores.append(1.25 * cur_tp / denom if denom > 0 else 0.0)

macro_f05 = np.mean(scores)
prec = tp_tot / max(1, tp_tot + fp_tot)
rec = tp_tot / max(1, tp_tot + fn_tot)

print("=" * 60)
print(f"STATE-AWARE CALIBRATED PRECISION & RECALL EVALUATION:")
print(f"Macro F_0.5 Score:     {macro_f05:.4f}")
print(f"Precision:             {prec:.4f} (TP: {tp_tot}, FP: {fp_tot})")
print(f"Recall:                {rec:.4f} (TP: {tp_tot}, FN: {fn_tot})")
print("=" * 60)
