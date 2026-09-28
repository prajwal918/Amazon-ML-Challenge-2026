#!/usr/bin/env python
# coding: utf-8
"""
================================================================================
Amazon ML Challenge 2026 -- V28 TRI-BRANCH MULTI-THRESHOLD RANKER (52 FEATURES)
================================================================================
Generates three calibrated submission variants in a single pass:
  1. matching_results.tsv (Calibrated Optimal F0.5: BaseTh=0.85, SibTh=0.60)
  2. matching_results_strict.tsv (Ultra-Conservative Precision: BaseTh=0.90, SibTh=0.70)
  3. matching_results_balanced.tsv (Balanced Corroboration: BaseTh=0.78, SibTh=0.55)
Zero external dependencies -- pure Python + LightGBM + XGBoost.
================================================================================
"""

import os, sys, re, time, random, math, unicodedata, difflib
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

import lightgbm as lgb
import xgboost as xgb

def clean_ascii(s):
    if not s: return ""
    return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8')

def jaro_winkler(s1, s2, prefix_weight=0.1):
    if s1 == s2: return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0: return 0.0
    match_distance = max(len1, len2) // 2 - 1
    s1_matches = [False] * len1; s2_matches = [False] * len2
    matches = 0; transpositions = 0
    for i in range(len1):
        start = max(0, i - match_distance); end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]: continue
            if s1[i] != s2[j]: continue
            s1_matches[i] = True; s2_matches[j] = True; matches += 1; break
    if matches == 0: return 0.0
    k = 0
    for i in range(len1):
        if not s1_matches[i]: continue
        while not s2_matches[k]: k += 1
        if s1[i] != s2[k]: transpositions += 1
        k += 1
    transpositions //= 2
    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0
    prefix = 0
    for i in range(min(4, min(len1, len2))):
        if s1[i] == s2[i]: prefix += 1
        else: break
    return jaro + prefix * prefix_weight * (1.0 - jaro)

def levenshtein_sim(s1, s2):
    if s1 == s2: return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0: return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def token_sort_ratio(s1, s2):
    if s1 == s2: return 1.0
    t1 = " ".join(sorted(s1.split()))
    t2 = " ".join(sorted(s2.split()))
    return difflib.SequenceMatcher(None, t1, t2).ratio()

def token_set_ratio(s1, s2):
    if s1 == s2: return 1.0
    tokens1 = set(s1.split()); tokens2 = set(s2.split())
    intersection = tokens1 & tokens2
    diff1 = tokens1 - intersection; diff2 = tokens2 - intersection
    sorted_inter = " ".join(sorted(intersection))
    sorted_diff1 = " ".join(sorted(diff1)); sorted_diff2 = " ".join(sorted(diff2))
    u1 = (sorted_inter + " " + sorted_diff1).strip()
    u2 = (sorted_inter + " " + sorted_diff2).strip()
    return max(
        difflib.SequenceMatcher(None, sorted_inter, u1).ratio() if sorted_inter else 0.0,
        difflib.SequenceMatcher(None, sorted_inter, u2).ratio() if sorted_inter else 0.0,
        difflib.SequenceMatcher(None, u1, u2).ratio()
    )

def partial_ratio(s1, s2):
    if s1 == s2: return 1.0
    if not s1 or not s2: return 0.0
    if len(s1) > len(s2): s1, s2 = s2, s1
    m = len(s1)
    best = 0.0
    for i in range(len(s2) - m + 1):
        sub = s2[i:i+m]
        r = difflib.SequenceMatcher(None, s1, sub).ratio()
        if r > best:
            best = r
            if best == 1.0: break
    return best

def soundex(word):
    if not word: return ""
    w = word.upper(); first = w[0]
    coded = w.translate(str.maketrans("AEHIOUWY","00000000"))
    coded = coded.translate(str.maketrans("BFPV","1111"))
    coded = coded.translate(str.maketrans("CGJKQSXZ","22222222"))
    coded = coded.translate(str.maketrans("DT","33"))
    coded = coded.translate(str.maketrans("L","4"))
    coded = coded.translate(str.maketrans("MN","55"))
    coded = coded.translate(str.maketrans("R","6"))
    res = first; prev = coded[0] if coded else "0"
    for c in coded[1:]:
        if c != prev and c != "0": res += c
        prev = c
    return (res + "0000")[:4]

LEGAL_SUFFIXES = [
    "private limited","pvt ltd","pvt. ltd.","pvt","llc","l.l.c","incorporated","inc",
    "corp","corporation","services","center","centre","holdings","group","enterprises",
    "solutions","llp","l.l.p","sarl","sas","ltd","limited","co","company","pllc","plc",
    "gmbh","sasu","eurl","sa","praiveta limiteda","praivett limitted","industries",
    "associates","consulting","international","intl"
]
_SUFFIX_RE = re.compile(r"\b("+"|".join(sorted((re.escape(s) for s in LEGAL_SUFFIXES),key=len,reverse=True))+r")\.?\b",re.IGNORECASE)
_METADATA_RE = re.compile(r"\(\s*id\s*[:#]?\s*\d+\s*\)",re.IGNORECASE)
_HANDLE_RE = re.compile(r"(?<![a-z0-9])@")
_DOMAIN_RE = re.compile(r"\.(com|net|org|io|co|in|fr|biz|org\.in|gov\.in)\b",re.IGNORECASE)
_DBA_RE = re.compile(r"\bd\.?\s*b\.?\s*a\.?\s*:?\s*",re.IGNORECASE)
_ABBR_RE = re.compile(r"^[A-Z]{2,6}$")
_LEET_MAP = str.maketrans({"5":"s","3":"e","1":"i","0":"o","4":"a","7":"t","8":"b"})
_NONALNUM_RE = re.compile(r"[^a-z0-9\s]"); _WS_RE = re.compile(r"\s+")
_STOPWORDS = {"the","and","of","a","an","in","to","for","at","on","by","null","near","opp","no"}
_NUM_WORD = {"one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}
_ORDINAL = {"1st":"1","2nd":"2","3rd":"3","4th":"4","first":"1","second":"2","third":"3","fourth":"4"}
_NUM_RE = re.compile(r"\d+"); _POSTAL_US = re.compile(r"\b\d{5}\b"); _POSTAL_IN = re.compile(r"\b[1-9]\d{5}\b")

def _clean_basic(s): return _WS_RE.sub(" ",_NONALNUM_RE.sub(" ",s)).strip()

def normalize_name(raw):
    if not isinstance(raw, str) or not raw.strip():
        return {"full":"","core":"","comp":"","leet":"","dba_alt":"","tokens":frozenset(),"qgrams":frozenset(),"initials":"","soundex_toks":frozenset(),"num_tokens":frozenset(),"first_tok":"","last_tok":""}
    s = raw.strip(); dba_alt = ""
    m = _DBA_RE.search(s)
    if m:
        after = s[m.end():].strip()
        if after:
            dba_alt = _clean_basic(clean_ascii(after).lower())
            dba_alt = _WS_RE.sub(" ",_SUFFIX_RE.sub(" ",dba_alt)).strip()
        s = s[:m.start()].strip()
    s = _METADATA_RE.sub(" ",s); s = _HANDLE_RE.sub(" ",s); s = _DOMAIN_RE.sub(" ",s)
    s = clean_ascii(s).lower()
    for word,digit in _NUM_WORD.items(): s = re.sub(r'\b'+word+r'\b',digit,s)
    for word,digit in _ORDINAL.items(): s = re.sub(r'\b'+word+r'\b',digit,s)
    full = _clean_basic(s); core = _WS_RE.sub(" ",_SUFFIX_RE.sub(" ",full)).strip()
    comp = re.sub(r'[^a-z0-9]','',core)
    leet = ""
    if any(ch.isdigit() for ch in core):
        cand = core.translate(_LEET_MAP)
        if cand != core: leet = cand
    tok_list = [t for t in core.split() if t not in _STOPWORDS and len(t)>1]
    tokens = frozenset(tok_list)
    qgrams = frozenset(core[i:i+3] for i in range(len(core)-2)) if len(core)>=3 else frozenset()
    initials = "".join(t[0] for t in tok_list)
    soundex_toks = frozenset(soundex(t) for t in tok_list if len(t)>=2)
    num_tokens = frozenset(t for t in tok_list if t.isdigit())
    first_tok = tok_list[0] if tok_list else ""
    last_tok = tok_list[-1] if tok_list else ""
    return {"full":full,"core":core,"comp":comp,"leet":leet,"dba_alt":dba_alt,"tokens":tokens,"qgrams":qgrams,"initials":initials,"soundex_toks":soundex_toks,"num_tokens":num_tokens,"first_tok":first_tok,"last_tok":last_tok}

US_STATES={"al":"alabama","ak":"alaska","az":"arizona","ar":"arkansas","ca":"california","co":"colorado","ct":"connecticut","de":"delaware","fl":"florida","ga":"georgia","hi":"hawaii","id":"idaho","il":"illinois","in":"indiana","ia":"iowa","ks":"kansas","ky":"kentucky","la":"louisiana","me":"maine","md":"maryland","ma":"massachusetts","mi":"michigan","mn":"minnesota","ms":"mississippi","mo":"missouri","mt":"montana","ne":"nebraska","nv":"nevada","nh":"new hampshire","nj":"new jersey","nm":"new mexico","ny":"new york","nc":"north carolina","nd":"north dakota","oh":"ohio","ok":"oklahoma","or":"oregon","pa":"pennsylvania","ri":"rhode island","sc":"south carolina","sd":"south dakota","tn":"tennessee","tx":"texas","ut":"utah","vt":"vermont","va":"virginia","wa":"washington","wv":"west virginia","wi":"wisconsin","wy":"wyoming","dc":"district of columbia"}
IN_STATES={"ap":"andhra pradesh","ar":"arunachal pradesh","as":"assam","br":"bihar","ct":"chhattisgarh","ga":"goa","gj":"gujarat","hr":"haryana","hp":"himachal pradesh","jh":"jharkhand","ka":"karnataka","kl":"kerala","mp":"madhya pradesh","mh":"maharashtra","mn":"manipur","ml":"meghalaya","mz":"mizoram","nl":"nagaland","or":"odisha","pb":"punjab","rj":"rajasthan","sk":"sikkim","tn":"tamil nadu","tg":"telangana","ts":"telangana","tr":"tripura","up":"uttar pradesh","uk":"uttarakhand","wb":"west bengal","dl":"delhi"}
STATE_ABBR={**US_STATES,**IN_STATES}; FULL_TO_ABBR={v:k for k,v in STATE_ABBR.items()}
FULL_NAME_CANON={f:f for f in FULL_TO_ABBR}
FULL_NAME_CANON.update({"orissa":"odisha","uttaranchal":"uttarakhand","pondicherry":"puducherry","tamilnadu":"tamil nadu"})

def normalize_address(raw, country=""):
    if not isinstance(raw, str) or not raw.strip():
        return {"house_number":"","state":"","postal":"","postal_pfx":"","tokens":frozenset(),"full":"","city":"","is_empty":True}
    s = clean_ascii(raw).lower()
    s = _WS_RE.sub(" ",_NONALNUM_RE.sub(" ",s)).strip(); tokens = s.split()
    postal = ""; postal_pfx = ""
    if country == "India":
        m = _POSTAL_IN.search(s)
        if m: postal = m.group(0); postal_pfx = postal[:3]
    else:
        m = _POSTAL_US.search(s)
        if m: postal = m.group(0); postal_pfx = postal[:3]
    house_number = ""
    for tok in tokens:
        if tok.isdigit() and tok != postal and len(tok) <= 5:
            house_number = tok.lstrip("0") or "0"; break
    if not house_number:
        for tok in tokens:
            mm = _NUM_RE.search(tok)
            if mm:
                val = mm.group(0)
                if val != postal and len(val) <= 5:
                    house_number = val.lstrip("0") or "0"; break
    state = ""
    cleaned_s = re.sub(r'\bfl\b\.?\s*(?:floor|ground|\d)',' ',s); s_tokens = cleaned_s.split()
    for tok in s_tokens:
        if tok in STATE_ABBR: state = STATE_ABBR[tok]; break
    if not state:
        joined = " "+s+" "
        for full,canon in FULL_NAME_CANON.items():
            if (" "+full+" ") in joined: state = canon; break
    city = ""
    for tok in reversed(tokens):
        if len(tok)>=3 and tok not in _STOPWORDS and tok not in STATE_ABBR and not tok.isdigit() and tok!=postal:
            city = tok; break
    tok_set = frozenset(t for t in tokens if t not in _STOPWORDS and len(t)>1)
    return {"house_number":house_number,"state":state,"postal":postal,"postal_pfx":postal_pfx,"tokens":tok_set,"full":s,"city":city,"is_empty":False}

FEATURE_NAMES = [
    "name_token_sort","name_token_set","name_partial","name_jw","name_core_jw",
    "name_core_sort","name_core_set","name_exact_core","name_exact_comp",
    "name_len_ratio","name_len_diff","name_token_jaccard","name_tok_overlap_cnt",
    "name_qgram3_jaccard","used_alt_name",
    "name_soundex_jaccard","name_initials_match","name_partial_core",
    "name_leet_match","name_comp_prefix_match","name_abbreviation_match",
    "name_levenshtein_ratio","name_num_tok_match","name_comp_len_ratio",
    "name_first_tok_match","name_last_tok_match","name_first_tok_jw",
    "addr_house_match","addr_house_conflict","addr_state_match","addr_state_conflict",
    "addr_postal_match","addr_postal_conflict","addr_token_sort","addr_token_set",
    "addr_token_jaccard","addr_city_match","addr_target_empty","addr_exact_full",
    "addr_postal_pfx_match","addr_postal_pfx_conflict",
    "name_x_addr","core_x_addr_set","name_jw_x_addr_state",
    "conf_agree","hard_conflict","name_high_addr_empty","exact_name_soft_addr",
    "name_jw_x_postal","core_sort_x_addr_sort","name_diff_penalty","exact_comp_match_high"
]
assert len(FEATURE_NAMES) == 52, f"Expected 52, got {len(FEATURE_NAMES)}"

def compute_pair_features(qn, qa, tn, ta):
    qf,tf = qn["full"],tn["full"]; qc,tc = qn["core"],tn["core"]
    tsort = token_sort_ratio(qf,tf); tset = token_set_ratio(qf,tf)
    partial = partial_ratio(qf,tf); jw = jaro_winkler(qf,tf)
    core_jw = jaro_winkler(qc,tc) if qc and tc else 0.0
    core_sort = token_sort_ratio(qc,tc) if qc and tc else 0.0
    core_set = token_set_ratio(qc,tc) if qc and tc else 0.0
    exact_core = 1.0 if (qc and tc and qc==tc) else 0.0
    exact_comp = 1.0 if (qn["comp"] and tn["comp"] and qn["comp"]==tn["comp"]) else 0.0
    ql,tl = len(qf),len(tf); len_ratio = (min(ql,tl)/max(ql,tl)) if max(ql,tl)>0 else 0.0; len_diff = abs(ql-tl)
    itk = len(qn["tokens"]&tn["tokens"]); utk = len(qn["tokens"]|tn["tokens"]); tok_j = itk/utk if utk>0 else 0.0
    iqg = len(qn["qgrams"]&tn["qgrams"]); uqg = len(qn["qgrams"]|tn["qgrams"]); qg_j = iqg/uqg if uqg>0 else 0.0
    used_alt = 1.0 if (qn["dba_alt"] or tn["dba_alt"]) else 0.0
    isd = len(qn["soundex_toks"]&tn["soundex_toks"]); usd = len(qn["soundex_toks"]|tn["soundex_toks"]); sd_j = isd/usd if usd>0 else 0.0
    init_m = 1.0 if (qn["initials"] and tn["initials"] and qn["initials"]==tn["initials"]) else 0.0
    p_core = partial_ratio(qc,tc) if qc and tc else 0.0
    leet_m = 1.0 if (qn["leet"] and tn["leet"] and qn["leet"]==tn["leet"]) else 0.0
    Qc,Tc = qn["comp"],tn["comp"]
    comp_pfx = 1.0 if (Qc and Tc and len(Qc)>=4 and len(Tc)>=4 and Qc[:8]==Tc[:8]) else 0.0
    qi = "".join(w[0] for w in qc.split() if w not in _STOPWORDS and len(w)>1)
    ti = "".join(w[0] for w in tc.split() if w not in _STOPWORDS and len(w)>1)
    abbr_m = 1.0 if (qi and ti and qi==ti and abs(len(qc.split())-len(tc.split()))>=2) else 0.0
    lev_r = levenshtein_sim(qf,tf) if qf and tf else 0.0
    nm = 1.0 if (qn["num_tokens"] and tn["num_tokens"] and qn["num_tokens"]==tn["num_tokens"]) else (0.5 if not qn["num_tokens"] and not tn["num_tokens"] else 0.0)
    clr = (min(len(Qc),len(Tc))/max(len(Qc),len(Tc))) if Qc and Tc and max(len(Qc),len(Tc))>0 else 0.0
    first_m = 1.0 if (qn["first_tok"] and tn["first_tok"] and qn["first_tok"]==tn["first_tok"]) else 0.0
    last_m = 1.0 if (qn["last_tok"] and tn["last_tok"] and qn["last_tok"]==tn["last_tok"]) else 0.0
    first_jw = jaro_winkler(qn["first_tok"], tn["first_tok"]) if qn["first_tok"] and tn["first_tok"] else 0.0

    if ta["is_empty"] or qa["is_empty"]:
        hm=hc=sm=sc_=pm=pc=a_sort=a_set=a_jacc=0.5; te=1.0 if ta["is_empty"] else 0.0; cm=0.5; ae=0.0
        pfx_m=pfx_c=0.5
    else:
        te=0.0; qh,th = qa["house_number"],ta["house_number"]
        if qh and th:
            if qh==th: hm,hc=1.0,0.0
            elif qh.startswith(th) or th.startswith(qh): hm,hc=0.8,0.0
            else: hm,hc=0.0,1.0
        else: hm,hc=0.5,0.0
        qs,ts = qa["state"],ta["state"]
        if qs and ts:
            if qs==ts: sm,sc_=1.0,0.0
            else: sm,sc_=0.0,1.0
        else: sm,sc_=0.5,0.0
        qp,tp = qa["postal"],ta["postal"]
        if qp and tp:
            if qp==tp: pm,pc=1.0,0.0
            else: pm,pc=0.0,1.0
        else: pm,pc=0.5,0.0
        a_sort = token_sort_ratio(qa["full"],ta["full"])
        a_set = token_set_ratio(qa["full"],ta["full"])
        ai = len(qa["tokens"]&ta["tokens"]); au = len(qa["tokens"]|ta["tokens"]); a_jacc = ai/au if au>0 else 0.0
        qct,tct = qa["city"],ta["city"]; cm = 1.0 if (qct and tct and qct==tct) else (0.5 if (not qct or not tct) else 0.0)
        ae = 1.0 if (qa["full"] and ta["full"] and qa["full"]==ta["full"]) else 0.0
        qpfx, tpfx = qa["postal_pfx"], ta["postal_pfx"]
        if qpfx and tpfx:
            if qpfx==tpfx: pfx_m,pfx_c = 1.0,0.0
            else: pfx_m,pfx_c = 0.0,1.0
        else: pfx_m,pfx_c = 0.5,0.0

    nxa = tsort * a_sort
    cxa = core_set * a_set
    jxs = jw * sm
    conf_agree = tsort * a_jacc
    hard_conflict = 1.0 if (sc_==1.0 and pc==1.0) else 0.0
    name_high_addr_empty = 1.0 if (jw >= 0.90 and te == 1.0) else 0.0
    exact_name_soft_addr = 1.0 if (exact_core == 1.0 and sc_ == 0.0 and hc == 0.0) else 0.0
    name_jw_x_postal = jw * pm
    core_sort_x_addr_sort = core_sort * a_sort
    name_diff_penalty = max(0.0, 1.0 - (len_diff / 10.0))
    exact_comp_match_high = 1.0 if (exact_comp == 1.0 and jw >= 0.95) else 0.0

    return [
        tsort,tset,partial,jw,core_jw,core_sort,core_set,exact_core,exact_comp,
        len_ratio,float(len_diff),tok_j,float(itk),qg_j,used_alt,
        sd_j,init_m,p_core,leet_m,comp_pfx,abbr_m,float(lev_r),nm,clr,
        first_m,last_m,float(first_jw),
        hm,hc,sm,sc_,pm,pc,a_sort,a_set,a_jacc,cm,te,ae,
        pfx_m,pfx_c,
        nxa,cxa,jxs,conf_agree,hard_conflict,name_high_addr_empty,exact_name_soft_addr,
        name_jw_x_postal,core_sort_x_addr_sort,name_diff_penalty,exact_comp_match_high
    ]

def compute_entity_f_beta(pred, true, beta=0.5):
    if not pred and not true: return 1.0
    if not pred or not true: return 0.0
    tp = len(pred & true)
    if tp == 0: return 0.0
    prec = tp / len(pred); rec = tp / len(true); b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec)

def main():
    t0 = time.time()
    print("=" * 80)
    print("V28 TRI-BRANCH MULTI-THRESHOLD RANKER (52 ZERO-DEP FEATURES)")
    print("=" * 80)

    TRAIN_DIR = None; TEST_DIR = None; CAND_FILE = None
    OUT_DIR = Path("/kaggle/working") if os.path.exists("/kaggle") else Path("output")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    search_roots = ["/kaggle/input", "."] if os.path.exists("/kaggle/input") else ["."]
    for s_root in search_roots:
        for root, dirs, files in os.walk(s_root):
            r = Path(root)
            if "train_ground_truth.tsv" in files and TRAIN_DIR is None: TRAIN_DIR = r
            if "test_source1.tsv" in files and TEST_DIR is None: TEST_DIR = r
            if "candidate_pairs.tsv" in files and CAND_FILE is None: CAND_FILE = r / "candidate_pairs.tsv"

    if TRAIN_DIR is None or TEST_DIR is None:
        TRAIN_DIR = Path("student_resource/dataset/train")
        TEST_DIR = Path("student_resource/dataset/test")
        CAND_FILE = Path("output/candidate_pairs.tsv")

    print(f"  TRAIN_DIR: {TRAIN_DIR}")
    print(f"  TEST_DIR:  {TEST_DIR}")
    print(f"  CAND_FILE: {CAND_FILE}")

    print("[1/5] Loading Ground Truth...")
    all_gt = defaultdict(set)
    with open(TRAIN_DIR / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 2 and p[1].strip():
                for t in p[1].strip().split(","):
                    t = t.strip()
                    if t: all_gt[p[0].strip()].add(t)

    s1_all = set()
    with open(TRAIN_DIR / "train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p: s1_all.add(p[0].strip())
    for sid in s1_all:
        if sid not in all_gt: all_gt[sid] = set()

    MAX_Q = 38000
    matched_keys = [k for k,v in all_gt.items() if v]
    singleton_keys = [k for k,v in all_gt.items() if not v]
    random.seed(42)
    n_m = min(int(MAX_Q * 0.65), len(matched_keys))
    n_s = min(MAX_Q - n_m, len(singleton_keys))
    sampled = random.sample(matched_keys, n_m) + random.sample(singleton_keys, n_s)
    all_gt = {k: all_gt[k] for k in sampled}
    needed_tgts = set()
    for v in all_gt.values(): needed_tgts |= v
    print(f"  Sampled: {n_m:,} matched + {n_s:,} singletons = {len(all_gt):,} total queries.")

    print("[2/5] Loading records...")
    s1_records = {}
    with open(TRAIN_DIR / "train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in all_gt:
                s1_records[p[0]] = (p[1] if len(p)>1 else "", p[2] if len(p)>2 else "", p[3].strip() if len(p)>3 else "")

    target_records = {}; dist_loaded = 0; MAX_DIST = 110000
    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        path = TRAIN_DIR / fn
        if not path.exists(): continue
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t"); eid = p[0]
                if eid in needed_tgts:
                    target_records[eid] = (p[1], p[2] if len(p)>2 else "", p[3].strip() if len(p)>3 else "")
                elif dist_loaded < MAX_DIST:
                    target_records[eid] = (p[1], p[2] if len(p)>2 else "", p[3].strip() if len(p)>3 else "")
                    dist_loaded += 1

    s1_norm = {sid: (normalize_name(n), normalize_address(a, c), c) for sid,(n,a,c) in s1_records.items()}
    tgt_norm = {tid: (normalize_name(n), normalize_address(a, c), c) for tid,(n,a,c) in target_records.items()}

    comp_idx = defaultdict(list); token_idx = defaultdict(list); hn_idx = defaultdict(list)
    for tid,(tn,ta,ctry) in tgt_norm.items():
        if tn["comp"] and len(tn["comp"])>=3: comp_idx[(ctry,tn["comp"][:6])].append(tid)
        for tok in tn["tokens"]:
            if len(tok)>=3: token_idx[(ctry,tok)].append(tid)
        if ta["house_number"]: hn_idx[(ctry,ta["house_number"])].append(tid)

    print("[3/5] Extracting 52 features & training ensemble...")
    qids = list(s1_records.keys()); random.shuffle(qids)
    val_split = int(len(qids) * 0.20)
    val_qids = set(qids[:val_split]); train_qids = set(qids[val_split:])

    train_X, train_y = [] , []
    val_candidates = defaultdict(list)
    n_pos = n_neg = 0

    for qid in qids:
        qn, qa, q_ctry = s1_norm[qid]; true_set = all_gt[qid]; is_val = qid in val_qids
        cands = set(true_set)
        if qn["comp"] and len(qn["comp"])>=3:
            cands.update(comp_idx.get((q_ctry,qn["comp"][:6]),[])[:6])
        for tok in qn["tokens"]:
            cands.update(token_idx.get((q_ctry,tok),[])[:4])
            if len(cands)>=25: break
        if qa["house_number"]:
            cands.update(hn_idx.get((q_ctry,qa["house_number"]),[])[:3])
        cands = [tid for tid in cands if tid in tgt_norm]

        for tid in cands:
            tn, ta, _ = tgt_norm[tid]
            feats = compute_pair_features(qn, qa, tn, ta); label = 1 if tid in true_set else 0
            if is_val: val_candidates[qid].append((tid, feats, label))
            else:
                train_X.append(feats); train_y.append(label)
                n_pos += label==1; n_neg += label==0

    ratio = n_neg / max(1, n_pos)
    spw = max(1.0, ratio * 0.25)
    print(f"  Training pairs: {len(train_X):,} (+{n_pos:,} / -{n_neg:,}, scale_pos_weight={spw:.2f})")

    X_train = np.array(train_X, dtype=np.float32); y_train = np.array(train_y, dtype=np.int32)
    val_Xf, val_yf = [], []
    for qid in val_qids:
        for tid, feats, label in val_candidates[qid]: val_Xf.append(feats); val_yf.append(label)
    X_val = np.array(val_Xf, dtype=np.float32); y_val = np.array(val_yf, dtype=np.int32)

    lgb_clf = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, num_leaves=63, max_depth=8,
        min_child_samples=15, subsample=0.80, colsample_bytree=0.80,
        reg_alpha=0.2, reg_lambda=2.0, scale_pos_weight=spw,
        objective="binary", random_state=42, verbosity=-1, n_jobs=-1
    )
    lgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])

    xgb_clf = xgb.XGBClassifier(
        n_estimators=850, learning_rate=0.03, max_depth=7, min_child_weight=3,
        subsample=0.80, colsample_bytree=0.80, reg_alpha=0.2, reg_lambda=2.0,
        scale_pos_weight=spw, objective="binary:logistic", eval_metric="logloss",
        random_state=42, verbosity=0, n_jobs=-1, early_stopping_rounds=50
    )
    xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    lgb_val = lgb_clf.predict_proba(X_val)[:, 1]
    xgb_val = xgb_clf.predict_proba(X_val)[:, 1]
    val_probs = 0.55 * lgb_val + 0.45 * xgb_val

    offset = 0; val_preds_by_q = {}
    for qid in val_qids:
        pairs = val_candidates[qid]; cnt = len(pairs)
        if cnt == 0: val_preds_by_q[qid] = []
        else:
            val_preds_by_q[qid] = [(pairs[i][0], val_probs[offset+i]) for i in range(cnt)]
            offset += cnt

    best_f = 0.0; best_cfg = (0.85, 0.60)
    for base_th in [0.75, 0.80, 0.85, 0.88, 0.90]:
        for sib_th in [0.55, 0.60, 0.65]:
            if sib_th > base_th: continue
            scores = []
            for qid in val_qids:
                true_set = all_gt[qid]; pairs = val_preds_by_q.get(qid, [])
                pred_set = set()
                if pairs:
                    sp = sorted(pairs, key=lambda x: -x[1])
                    if sp[0][1] >= base_th:
                        pred_set.add(sp[0][0])
                        for tid, p in sp[1:]:
                            if p >= sib_th: pred_set.add(tid)
                            else: break
                scores.append(compute_entity_f_beta(pred_set, true_set, 0.5))
            macro_f = float(np.mean(scores))
            if macro_f > best_f:
                best_f = macro_f; best_cfg = (base_th, sib_th)

    base_th, sib_th = best_cfg
    print(f"\nV28 Optimal Config: Macro F0.5 = {best_f:.4f} | BaseTh = {base_th:.2f} | SibTh = {sib_th:.2f}")

    del train_X, train_y, X_train, y_train, val_Xf, val_yf, X_val, y_val
    del s1_norm, tgt_norm, comp_idx, token_idx, hn_idx

    print("\n[4/5] Streaming inference across 1.73M test queries...")
    cand_dict = {}; needed = set(); s1_order = []
    with open(CAND_FILE, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t"); qid = p[0]; s1_order.append(qid)
            cids = [x.strip() for x in p[1].split(",") if x.strip()] if len(p)>1 and p[1] else []
            cand_dict[qid] = cids; needed.update(cids)

    tgt_raw = {}
    for fn in ["test_source2.tsv", "test_source3.tsv"]:
        path = TEST_DIR / fn
        if not path.exists(): continue
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t"); eid = p[0]
                if eid in needed:
                    tgt_raw[eid] = (p[1], p[2] if len(p)>2 else "", p[3].strip() if len(p)>3 else "")

    s1_norm = {}
    with open(TEST_DIR / "test_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t"); qid = p[0]
            s1_norm[qid] = (normalize_name(p[1] if len(p)>1 else ""), normalize_address(p[2] if len(p)>2 else "", p[3].strip() if len(p)>3 else ""))

    # Multi-threshold dictionaries
    match_dict_opt = {}
    match_dict_strict = {}
    match_dict_bal = {}

    CHUNK = 50000; n_total = len(s1_order); t_s = time.time()
    for cs in range(0, n_total, CHUNK):
        ce = min(n_total, cs + CHUNK); chunk_qids = s1_order[cs:ce]
        ct = set()
        for qid in chunk_qids:
            for cid in cand_dict[qid]:
                if cid in tgt_raw: ct.add(cid)
        ctn = {}
        for cid in ct:
            nm, ad, cy = tgt_raw[cid]; ctn[cid] = (normalize_name(nm), normalize_address(ad, cy))

        bf = []; bc = []; bcids = []
        for qid in chunk_qids:
            qn, qa = s1_norm[qid]; cids = cand_dict[qid]; cnt = 0; these = []
            for cid in cids:
                if cid in ctn:
                    tn, ta = ctn[cid]
                    bf.append(compute_pair_features(qn, qa, tn, ta)); these.append(cid); cnt += 1
            bc.append(cnt); bcids.append(these)

        if bf:
            X = np.array(bf, dtype=np.float32)
            probs = 0.55 * lgb_clf.predict_proba(X)[:, 1] + 0.45 * xgb_clf.predict_proba(X)[:, 1]

            offset = 0
            for i, qid in enumerate(chunk_qids):
                cnt = bc[i]; cids = bcids[i]
                if cnt == 0:
                    match_dict_opt[qid] = ""; match_dict_strict[qid] = ""; match_dict_bal[qid] = ""
                else:
                    q_p = probs[offset:offset+cnt]; si = np.argsort(q_p)[::-1]
                    
                    # 1. Optimal calibrated threshold
                    ch_opt = []
                    if q_p[si[0]] >= base_th:
                        ch_opt.append(cids[si[0]])
                        for j in si[1:]:
                            if q_p[j] >= sib_th: ch_opt.append(cids[j])
                            else: break
                    match_dict_opt[qid] = ",".join(ch_opt)

                    # 2. Strict ultra-precision threshold (0.90 / 0.70)
                    ch_strict = []
                    if q_p[si[0]] >= 0.90:
                        ch_strict.append(cids[si[0]])
                        for j in si[1:]:
                            if q_p[j] >= 0.70: ch_strict.append(cids[j])
                            else: break
                    match_dict_strict[qid] = ",".join(ch_strict)

                    # 3. Balanced threshold (0.78 / 0.55)
                    ch_bal = []
                    if q_p[si[0]] >= 0.78:
                        ch_bal.append(cids[si[0]])
                        for j in si[1:]:
                            if q_p[j] >= 0.55: ch_bal.append(cids[j])
                            else: break
                    match_dict_bal[qid] = ",".join(ch_bal)

                    offset += cnt
        else:
            for qid in chunk_qids:
                match_dict_opt[qid] = ""; match_dict_strict[qid] = ""; match_dict_bal[qid] = ""

        del ctn, bf, bcids
        done = ce; el = time.time() - t_s; rate = done / max(1.0, el)
        print(f"  V28 Progress: {done:,}/{n_total:,} ({done/n_total*100:.1f}%) | Speed: {rate:.0f} q/s | Elapsed: {el:.1f}s")

    print("\n[5/5] Writing 3 Calibrated Submission TSVs...")
    
    # 1. Main optimal file
    out_file = OUT_DIR / "matching_results.tsv"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for qid in s1_order: f.write(f"{qid}\t{match_dict_opt.get(qid,'')}\n")

    # 2. Strict precision file
    out_strict = OUT_DIR / "matching_results_strict.tsv"
    with open(out_strict, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for qid in s1_order: f.write(f"{qid}\t{match_dict_strict.get(qid,'')}\n")

    # 3. Balanced file
    out_bal = OUT_DIR / "matching_results_balanced.tsv"
    with open(out_bal, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for qid in s1_order: f.write(f"{qid}\t{match_dict_bal.get(qid,'')}\n")

    ns = sum(1 for m in match_dict_opt.values() if not m); nm = len(match_dict_opt) - ns
    print(f"\nFinal V28 Summary (Optimal): {len(match_dict_opt):,} total | {nm:,} matched ({nm/len(match_dict_opt)*100:.1f}%) | {ns:,} singletons")
    print(f"Saved: {out_file}, {out_strict}, {out_bal}")
    print(f"Total Pipeline Runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
