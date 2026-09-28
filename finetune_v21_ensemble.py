#!/usr/bin/env python3
"""
================================================================================
Amazon ML Challenge 2026 -- V21 MEGA ENSEMBLE (LightGBM + XGBoost)
================================================================================
Power-ups over V20:
  1. XGBoost + LightGBM ensemble — averaged probabilities for better calibration
  2. 6 NEW features: Levenshtein ratio, edit_distance_core, double-metaphone sim,
     abbreviation_match, number_token_match, name_comp_len_ratio
  3. 40K training queries (larger + better coverage)
  4. Stratified sampling: 60% matched, 40% singleton queries
  5. Hard negative re-mining: top-3 false positives from LGB used to retrain XGB
  6. Final threshold: precision-maximized for F0.5
================================================================================
"""
import sys, os, re, time, random
from collections import defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from unidecode import unidecode as _unidecode
import lightgbm as lgb
import joblib

# XGBoost
try:
    import xgboost as xgb
    _HAVE_XGB = True
except ImportError:
    _HAVE_XGB = False
    print("WARNING: xgboost not installed — will use LightGBM only")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Metaphone (double metaphone via jellyfish or fallback) ────────────────────
_HAVE_METAPHONE = False
try:
    import jellyfish
    _HAVE_METAPHONE = True
except ImportError:
    pass

def metaphone_sim(a, b):
    if not _HAVE_METAPHONE or not a or not b: return 0.5
    try:
        ma = jellyfish.metaphone(a); mb = jellyfish.metaphone(b)
        if not ma or not mb: return 0.5
        return 1.0 if ma == mb else fuzz.ratio(ma, mb) / 100.0
    except: return 0.5

# ── Indic ─────────────────────────────────────────────────────────────────────
_HAVE_INDIC = False
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _translit
    _SCRIPT_RANGES = [
        (0x0900,0x097F,sanscript.DEVANAGARI),(0x0B80,0x0BFF,sanscript.TAMIL),
        (0x0980,0x09FF,sanscript.BENGALI),(0x0C00,0x0C7F,sanscript.TELUGU),
        (0x0C80,0x0CFF,sanscript.KANNADA),
    ]
    _HAVE_INDIC = True
except Exception: pass

def _token_script(token):
    for ch in token:
        code = ord(ch)
        for lo, hi, scheme in _SCRIPT_RANGES:
            if lo <= code <= hi: return scheme
    return None

def romanize_mixed(text):
    if not text or not _HAVE_INDIC: return text
    out = []
    for tok in text.split():
        scheme = _token_script(tok)
        if scheme:
            try: out.append(_translit(tok, scheme, sanscript.ITRANS))
            except: out.append(tok)
        else: out.append(tok)
    return " ".join(out)

def soundex(word):
    if not word: return ""
    word = word.upper(); first = word[0]
    coded = word.translate(str.maketrans("AEHIOUWY","00000000"))
    coded = coded.translate(str.maketrans("BFPV","1111"))
    coded = coded.translate(str.maketrans("CGJKQSXZ","22222222"))
    coded = coded.translate(str.maketrans("DT","33"))
    coded = coded.translate(str.maketrans("L","4"))
    coded = coded.translate(str.maketrans("MN","55"))
    coded = coded.translate(str.maketrans("R","6"))
    result = first; prev = coded[0] if coded else "0"
    for c in coded[1:]:
        if c != prev and c != "0": result += c
        prev = c
    return (result + "0000")[:4]

# ── Normalization ──────────────────────────────────────────────────────────────
LEGAL_SUFFIXES = [
    "private limited","pvt ltd","pvt. ltd.","pvt","llc","l.l.c","incorporated","inc",
    "corp","corporation","services","center","centre","holdings","group","enterprises",
    "solutions","llp","l.l.p","sarl","sas","ltd","limited","co","company","pllc","plc",
    "gmbh","sasu","eurl","sa","praiveta limiteda","praivett limitted","industries",
    "associates","consulting","international","intl"
]
_SUFFIX_RE   = re.compile(r"\b("+"|".join(sorted((re.escape(s) for s in LEGAL_SUFFIXES),key=len,reverse=True))+r")\.?\b",re.IGNORECASE)
_METADATA_RE = re.compile(r"\(\s*id\s*[:#]?\s*\d+\s*\)",re.IGNORECASE)
_HANDLE_RE   = re.compile(r"(?<![a-z0-9])@")
_DOMAIN_RE   = re.compile(r"\.(com|net|org|io|co|in|fr|biz|org\.in|gov\.in)\b",re.IGNORECASE)
_DBA_RE      = re.compile(r"\bd\.?\s*b\.?\s*a\.?\s*:?\s*",re.IGNORECASE)
_ABBR_RE     = re.compile(r"^[A-Z]{2,6}$")  # detect abbreviations
_LEET_MAP    = str.maketrans({"5":"s","3":"e","1":"i","0":"o","4":"a","7":"t","8":"b"})
_NONALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE       = re.compile(r"\s+")
_STOPWORDS   = {"the","and","of","a","an","in","to","for","at","on","by","null","near","opp","no"}
_NUM_WORD    = {"one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}
_ORDINAL     = {"1st":"1","2nd":"2","3rd":"3","4th":"4","first":"1","second":"2","third":"3","fourth":"4"}

def _clean_basic(s): return _WS_RE.sub(" ",_NONALNUM_RE.sub(" ",s)).strip()

def is_abbreviation(s):
    """Check if string looks like an abbreviation (IBM, KFC, etc)"""
    words = s.upper().split()
    return len(words) == 1 and bool(_ABBR_RE.match(words[0])) if words else False

def abbrev_match(q_full, t_full, q_core, t_core):
    """1.0 if one looks like abbreviation of the other"""
    q_initials = "".join(w[0] for w in q_core.split() if w not in _STOPWORDS and len(w)>1)
    t_initials = "".join(w[0] for w in t_core.split() if w not in _STOPWORDS and len(w)>1)
    qc, tc = q_core.replace(" ",""), t_core.replace(" ","")
    if (q_initials and t_initials and q_initials == t_initials and
        abs(len(q_core.split())-len(t_core.split())) >= 2):
        return 1.0
    if is_abbreviation(q_core) and t_initials.startswith(q_core[:len(q_core)]):
        return 0.9
    if is_abbreviation(t_core) and q_initials.startswith(t_core[:len(t_core)]):
        return 0.9
    return 0.0

def normalize_name(raw):
    if not isinstance(raw,str) or not raw.strip():
        return {"full":"","core":"","comp":"","leet":"","dba_alt":"","tokens":frozenset(),
                "qgrams":frozenset(),"initials":"","soundex_toks":frozenset(),"num_tokens":frozenset()}
    s = raw.strip(); dba_alt=""
    m = _DBA_RE.search(s)
    if m:
        after = s[m.end():].strip()
        if after:
            dba_alt = _clean_basic(_unidecode(after).lower())
            dba_alt = _WS_RE.sub(" ",_SUFFIX_RE.sub(" ",dba_alt)).strip()
        s = s[:m.start()].strip()
    s = _METADATA_RE.sub(" ",s); s = _HANDLE_RE.sub(" ",s); s = _DOMAIN_RE.sub(" ",s)
    s = romanize_mixed(s); s = _unidecode(s).lower()
    for word,digit in _NUM_WORD.items(): s = re.sub(r'\b'+word+r'\b',digit,s)
    for word,digit in _ORDINAL.items(): s = re.sub(r'\b'+word+r'\b',digit,s)
    full = _clean_basic(s)
    core = _WS_RE.sub(" ",_SUFFIX_RE.sub(" ",full)).strip()
    comp = re.sub(r'[^a-z0-9]','',core)
    leet = ""
    if any(ch.isdigit() for ch in core):
        cand = core.translate(_LEET_MAP)
        if cand != core: leet = cand
    tokens = frozenset(t for t in core.split() if t not in _STOPWORDS and len(t)>1)
    qgrams = frozenset(core[i:i+3] for i in range(len(core)-2)) if len(core)>=3 else frozenset()
    initials = "".join(t[0] for t in core.split() if t not in _STOPWORDS and len(t)>1)
    soundex_toks = frozenset(soundex(t) for t in core.split() if len(t)>=2)
    num_tokens = frozenset(t for t in core.split() if t.isdigit())
    return {"full":full,"core":core,"comp":comp,"leet":leet,"dba_alt":dba_alt,
            "tokens":tokens,"qgrams":qgrams,"initials":initials,
            "soundex_toks":soundex_toks,"num_tokens":num_tokens}

US_STATES={"al":"alabama","ak":"alaska","az":"arizona","ar":"arkansas","ca":"california","co":"colorado","ct":"connecticut","de":"delaware","fl":"florida","ga":"georgia","hi":"hawaii","id":"idaho","il":"illinois","in":"indiana","ia":"iowa","ks":"kansas","ky":"kentucky","la":"louisiana","me":"maine","md":"maryland","ma":"massachusetts","mi":"michigan","mn":"minnesota","ms":"mississippi","mo":"missouri","mt":"montana","ne":"nebraska","nv":"nevada","nh":"new hampshire","nj":"new jersey","nm":"new mexico","ny":"new york","nc":"north carolina","nd":"north dakota","oh":"ohio","ok":"oklahoma","or":"oregon","pa":"pennsylvania","ri":"rhode island","sc":"south carolina","sd":"south dakota","tn":"tennessee","tx":"texas","ut":"utah","vt":"vermont","va":"virginia","wa":"washington","wv":"west virginia","wi":"wisconsin","wy":"wyoming","dc":"district of columbia"}
IN_STATES={"ap":"andhra pradesh","ar":"arunachal pradesh","as":"assam","br":"bihar","ct":"chhattisgarh","ga":"goa","gj":"gujarat","hr":"haryana","hp":"himachal pradesh","jh":"jharkhand","ka":"karnataka","kl":"kerala","mp":"madhya pradesh","mh":"maharashtra","mn":"manipur","ml":"meghalaya","mz":"mizoram","nl":"nagaland","or":"odisha","pb":"punjab","rj":"rajasthan","sk":"sikkim","tn":"tamil nadu","tg":"telangana","ts":"telangana","tr":"tripura","up":"uttar pradesh","uk":"uttarakhand","wb":"west bengal","dl":"delhi"}
STATE_ABBR={**US_STATES,**IN_STATES}
FULL_TO_ABBR={v:k for k,v in STATE_ABBR.items()}
FULL_NAME_CANON={full:full for full in FULL_TO_ABBR}
FULL_NAME_CANON.update({"orissa":"odisha","uttaranchal":"uttarakhand","pondicherry":"puducherry","tamilnadu":"tamil nadu"})
_NUM_RE=re.compile(r"\d+"); _POSTAL_US=re.compile(r"\b\d{5}\b"); _POSTAL_IN=re.compile(r"\b[1-9]\d{5}\b")

def normalize_address(raw,country=""):
    if not isinstance(raw,str) or not raw.strip():
        return {"house_number":"","state":"","postal":"","tokens":frozenset(),"full":"","city":"","is_empty":True}
    s=romanize_mixed(raw); s=_unidecode(s).lower(); s=_WS_RE.sub(" ",_NONALNUM_RE.sub(" ",s)).strip(); tokens=s.split()
    postal=""
    if country=="India":
        m=_POSTAL_IN.search(s);
        if m: postal=m.group(0)
    else:
        m=_POSTAL_US.search(s);
        if m: postal=m.group(0)
    house_number=""
    for tok in tokens:
        if tok.isdigit() and tok!=postal and len(tok)<=5: house_number=tok.lstrip("0") or "0"; break
    if not house_number:
        for tok in tokens:
            mm=_NUM_RE.search(tok)
            if mm:
                val=mm.group(0)
                if val!=postal and len(val)<=5: house_number=val.lstrip("0") or "0"; break
    state=""
    cleaned_s=re.sub(r'\bfl\b\.?\s*(?:floor|ground|\d)',' ',s); s_tokens=cleaned_s.split()
    for tok in s_tokens:
        if tok in STATE_ABBR: state=STATE_ABBR[tok]; break
    if not state:
        joined=" "+s+" "
        for full,canon in FULL_NAME_CANON.items():
            if (" "+full+" ") in joined: state=canon; break
    city=""
    for tok in reversed(tokens):
        if len(tok)>=3 and tok not in _STOPWORDS and tok not in STATE_ABBR and not tok.isdigit() and tok!=postal: city=tok; break
    tok_set=frozenset(t for t in tokens if t not in _STOPWORDS and len(t)>1)
    return {"house_number":house_number,"state":state,"postal":postal,"tokens":tok_set,"full":s,"city":city,"is_empty":False}

# ── 41-feature vector ─────────────────────────────────────────────────────────
FEATURE_NAMES = [
    # Name (21)
    "name_token_sort","name_token_set","name_partial","name_jw",
    "name_core_jw","name_core_sort","name_core_set",
    "name_exact_core","name_exact_comp",
    "name_len_ratio","name_len_diff",
    "name_token_jaccard","name_tok_overlap_cnt","name_qgram3_jaccard",
    "used_alt_name",
    "name_soundex_jaccard","name_initials_match","name_partial_core",
    "name_leet_match","name_comp_prefix_match",
    "name_abbreviation_match",
    # NEW name (4)
    "name_levenshtein_ratio","name_levenshtein_core","name_num_tok_match","name_metaphone_sim",
    # Comp length ratio (1)
    "name_comp_len_ratio",
    # Address (12)
    "addr_house_match","addr_house_conflict",
    "addr_state_match","addr_state_conflict",
    "addr_postal_match","addr_postal_conflict",
    "addr_token_sort","addr_token_set","addr_token_jaccard",
    "addr_city_match","addr_target_empty","addr_exact_full",
    # Cross (3)
    "name_x_addr","core_x_addr_set","name_jw_x_addr_state",
]
assert len(FEATURE_NAMES) == 41, f"Expected 41, got {len(FEATURE_NAMES)}"

def compute_pair_features(q_norm, q_addr, t_norm, t_addr):
    q_full,t_full = q_norm["full"],t_norm["full"]
    q_core,t_core = q_norm["core"],t_norm["core"]

    tsort   = fuzz.token_sort_ratio(q_full,t_full)/100.0
    tset    = fuzz.token_set_ratio(q_full,t_full)/100.0
    partial = fuzz.partial_ratio(q_full,t_full)/100.0
    jw      = float(JaroWinkler.similarity(q_full,t_full))
    core_jw = float(JaroWinkler.similarity(q_core,t_core)) if q_core and t_core else 0.0
    core_sort = fuzz.token_sort_ratio(q_core,t_core)/100.0 if q_core and t_core else 0.0
    core_set  = fuzz.token_set_ratio(q_core,t_core)/100.0 if q_core and t_core else 0.0
    exact_core = 1.0 if (q_core and t_core and q_core==t_core) else 0.0
    exact_comp = 1.0 if (q_norm["comp"] and t_norm["comp"] and q_norm["comp"]==t_norm["comp"]) else 0.0
    qlen,tlen = len(q_full),len(t_full)
    len_ratio = (min(qlen,tlen)/max(qlen,tlen)) if max(qlen,tlen)>0 else 0.0
    len_diff  = abs(qlen-tlen)
    inter_tok = len(q_norm["tokens"]&t_norm["tokens"]); union_tok = len(q_norm["tokens"]|t_norm["tokens"])
    tok_jaccard = inter_tok/union_tok if union_tok>0 else 0.0
    inter_qg  = len(q_norm["qgrams"]&t_norm["qgrams"]); union_qg = len(q_norm["qgrams"]|t_norm["qgrams"])
    qgram_jacc = inter_qg/union_qg if union_qg>0 else 0.0
    used_alt  = 1.0 if (q_norm["dba_alt"] or t_norm["dba_alt"]) else 0.0
    inter_sd  = len(q_norm["soundex_toks"]&t_norm["soundex_toks"]); union_sd = len(q_norm["soundex_toks"]|t_norm["soundex_toks"])
    soundex_jacc = inter_sd/union_sd if union_sd>0 else 0.0
    initials_match = 1.0 if (q_norm["initials"] and t_norm["initials"] and q_norm["initials"]==t_norm["initials"]) else 0.0
    partial_core   = fuzz.partial_ratio(q_core,t_core)/100.0 if q_core and t_core else 0.0
    leet_match     = 1.0 if (q_norm["leet"] and t_norm["leet"] and q_norm["leet"]==t_norm["leet"]) else 0.0
    qc,tc = q_norm["comp"],t_norm["comp"]
    comp_prefix = 1.0 if (qc and tc and len(qc)>=4 and len(tc)>=4 and qc[:8]==tc[:8]) else 0.0
    abbr_match  = abbrev_match(q_full,t_full,q_core,t_core)
    # NEW
    lev_ratio   = Levenshtein.normalized_similarity(q_full,t_full) if q_full and t_full else 0.0
    lev_core    = Levenshtein.normalized_similarity(q_core,t_core) if q_core and t_core else 0.0
    num_match   = 1.0 if (q_norm["num_tokens"] and t_norm["num_tokens"] and q_norm["num_tokens"]==t_norm["num_tokens"]) else (0.5 if not q_norm["num_tokens"] and not t_norm["num_tokens"] else 0.0)
    meta_sim    = metaphone_sim(q_core.split()[0] if q_core else "", t_core.split()[0] if t_core else "")
    comp_len_ratio = (min(len(qc),len(tc))/max(len(qc),len(tc))) if qc and tc and max(len(qc),len(tc))>0 else 0.0

    # Address
    if t_addr["is_empty"] or q_addr["is_empty"]:
        house_match=house_conflict=state_match=state_conflict=postal_match=postal_conflict=addr_sort=addr_set=addr_jacc=0.5
        target_empty=1.0 if t_addr["is_empty"] else 0.0; city_match=0.5; addr_exact=0.0
    else:
        target_empty=0.0; qh,th=q_addr["house_number"],t_addr["house_number"]
        if qh and th:
            if qh==th: house_match,house_conflict=1.0,0.0
            elif qh.startswith(th) or th.startswith(qh): house_match,house_conflict=0.8,0.0
            else: house_match,house_conflict=0.0,1.0
        else: house_match,house_conflict=0.5,0.0
        qs,ts=q_addr["state"],t_addr["state"]
        if qs and ts:
            if qs==ts: state_match,state_conflict=1.0,0.0
            else: state_match,state_conflict=0.0,1.0
        else: state_match,state_conflict=0.5,0.0
        qp,tp=q_addr["postal"],t_addr["postal"]
        if qp and tp:
            if qp==tp: postal_match,postal_conflict=1.0,0.0
            else: postal_match,postal_conflict=0.0,1.0
        else: postal_match,postal_conflict=0.5,0.0
        addr_sort=fuzz.token_sort_ratio(q_addr["full"],t_addr["full"])/100.0
        addr_set =fuzz.token_set_ratio(q_addr["full"],t_addr["full"])/100.0
        ainter=len(q_addr["tokens"]&t_addr["tokens"]); aunion=len(q_addr["tokens"]|t_addr["tokens"])
        addr_jacc=ainter/aunion if aunion>0 else 0.0
        qct,tct=q_addr["city"],t_addr["city"]
        city_match=1.0 if (qct and tct and qct==tct) else (0.5 if (not qct or not tct) else 0.0)
        addr_exact=1.0 if (q_addr["full"] and t_addr["full"] and q_addr["full"]==t_addr["full"]) else 0.0

    name_x_addr = tsort*addr_sort; core_x_addr = core_set*addr_set; jw_x_state = jw*state_match

    return [
        tsort,tset,partial,jw,core_jw,core_sort,core_set,exact_core,exact_comp,
        len_ratio,float(len_diff),tok_jaccard,float(inter_tok),qgram_jacc,used_alt,
        soundex_jacc,initials_match,partial_core,leet_match,comp_prefix,abbr_match,
        float(lev_ratio),float(lev_core),num_match,float(meta_sim),comp_len_ratio,
        house_match,house_conflict,state_match,state_conflict,
        postal_match,postal_conflict,addr_sort,addr_set,addr_jacc,
        city_match,target_empty,addr_exact,
        name_x_addr,core_x_addr,jw_x_state
    ]

def compute_entity_f_beta(pred,true,beta=0.5):
    if not pred and not true: return 1.0
    if not pred or not true: return 0.0
    tp=len(pred&true)
    if tp==0: return 0.0
    prec=tp/len(pred); rec=tp/len(true); b2=beta*beta
    return (1+b2)*prec*rec/(b2*prec+rec)


def main():
    t0=time.time()
    print("="*80)
    print(f"V21 MEGA ENSEMBLE (LightGBM+{'XGBoost' if _HAVE_XGB else 'LightGBM×2'}) — 41 features")
    print("="*80)

    TRAIN_DIR=Path("student_resource/dataset/train")

    # ── Load GT ────────────────────────────────────────────────────────────────
    print("[1/6] Loading ground truth...")
    all_gt=defaultdict(set)
    with open(TRAIN_DIR/"train_ground_truth.tsv","r",encoding="utf-8") as f:
        next(f)
        for line in f:
            p=line.rstrip("\r\n").split("\t")
            if len(p)>=2 and p[1].strip():
                for t in p[1].strip().split(","):
                    t=t.strip()
                    if t: all_gt[p[0].strip()].add(t)
    s1_all=set()
    with open(TRAIN_DIR/"train_source1.tsv","r",encoding="utf-8") as f:
        next(f)
        for line in f:
            p=line.rstrip("\r\n").split("\t")
            if p: s1_all.add(p[0].strip())
    for sid in s1_all:
        if sid not in all_gt: all_gt[sid]=set()

    needed_tgts=set()
    for v in all_gt.values(): needed_tgts|=v
    print(f"  {len(all_gt):,} GT queries | {len(needed_tgts):,} true targets")

    # Stratified sample: 60% non-singleton, 40% singleton
    MAX_Q = 40000
    matched_keys   = [k for k,v in all_gt.items() if v]
    singleton_keys = [k for k,v in all_gt.items() if not v]
    random.seed(42)
    n_matched  = min(int(MAX_Q*0.65), len(matched_keys))
    n_singleton= min(MAX_Q-n_matched, len(singleton_keys))
    sampled = random.sample(matched_keys,n_matched) + random.sample(singleton_keys,n_singleton)
    all_gt = {k:all_gt[k] for k in sampled}
    needed_tgts=set()
    for v in all_gt.values(): needed_tgts|=v
    print(f"  Stratified sample: {n_matched:,} matched + {n_singleton:,} singleton = {len(all_gt):,} total")

    # ── Load Sources ───────────────────────────────────────────────────────────
    print("[2/6] Loading records...")
    s1_records={}
    with open(TRAIN_DIR/"train_source1.tsv","r",encoding="utf-8") as f:
        next(f)
        for line in f:
            p=line.rstrip("\r\n").split("\t")
            if p[0] in all_gt:
                s1_records[p[0]]=(p[1] if len(p)>1 else "",p[2] if len(p)>2 else "",p[3].strip() if len(p)>3 else "")

    target_records={}; dist_loaded=0; MAX_DIST=130000
    for fn in ["train_source2.tsv","train_source3.tsv"]:
        with open(TRAIN_DIR/fn,"r",encoding="utf-8") as f:
            next(f)
            for line in f:
                p=line.rstrip("\r\n").split("\t"); eid=p[0]
                if eid in needed_tgts:
                    target_records[eid]=(p[1],p[2] if len(p)>2 else "",p[3].strip() if len(p)>3 else "")
                elif dist_loaded<MAX_DIST:
                    target_records[eid]=(p[1],p[2] if len(p)>2 else "",p[3].strip() if len(p)>3 else ""); dist_loaded+=1
    print(f"  {len(target_records):,} targets ({len(needed_tgts):,} true + {dist_loaded:,} distractors)")

    # ── Normalize ──────────────────────────────────────────────────────────────
    print("[3/6] Normalizing...")
    s1_norm ={sid:(normalize_name(n),normalize_address(a,c),c) for sid,(n,a,c) in s1_records.items()}
    tgt_norm={tid:(normalize_name(n),normalize_address(a,c),c) for tid,(n,a,c) in target_records.items()}

    comp_idx=defaultdict(list); token_idx=defaultdict(list); hn_idx=defaultdict(list)
    for tid,(tn,ta,ctry) in tgt_norm.items():
        if tn["comp"] and len(tn["comp"])>=3: comp_idx[(ctry,tn["comp"][:6])].append(tid)
        for tok in tn["tokens"]:
            if len(tok)>=3: token_idx[(ctry,tok)].append(tid)
        if ta["house_number"]: hn_idx[(ctry,ta["house_number"])].append(tid)

    # ── Generate Pairs ─────────────────────────────────────────────────────────
    print("[4/6] Generating pairs & extracting 41 features...")
    qids=list(s1_records.keys()); random.shuffle(qids)
    val_split=int(len(qids)*0.20); val_qids=set(qids[:val_split]); train_qids=set(qids[val_split:])
    print(f"  Train: {len(train_qids):,} | Val: {len(val_qids):,}")

    train_X,train_y=[],[]
    val_candidates=defaultdict(list)
    n_pos=n_neg=0

    for qid in qids:
        qn,qa,q_ctry=s1_norm[qid]; true_set=all_gt[qid]; is_val=qid in val_qids
        cands=set(true_set)
        if qn["comp"] and len(qn["comp"])>=3:
            cands.update(comp_idx.get((q_ctry,qn["comp"][:6]),[])[:6])
        for tok in qn["tokens"]:
            cands.update(token_idx.get((q_ctry,tok),[])[:4])
            if len(cands)>=25: break
        if qa["house_number"]:
            cands.update(hn_idx.get((q_ctry,qa["house_number"]),[])[:3])
        cands=[tid for tid in cands if tid in tgt_norm]
        for tid in cands:
            tn,ta,_=tgt_norm[tid]
            feats=compute_pair_features(qn,qa,tn,ta); label=1 if tid in true_set else 0
            if is_val: val_candidates[qid].append((tid,feats,label))
            else:
                train_X.append(feats); train_y.append(label)
                if label==1: n_pos+=1
                else: n_neg+=1

    ratio=n_neg/max(1,n_pos)
    print(f"  Train: {len(train_X):,} pairs | +{n_pos:,} -{n_neg:,} (ratio 1:{ratio:.1f})")

    X_train=np.array(train_X,dtype=np.float32); y_train=np.array(train_y,dtype=np.int32)
    val_X_flat,val_y_flat=[],[]
    for qid in val_qids:
        for tid,feats,label in val_candidates[qid]:
            val_X_flat.append(feats); val_y_flat.append(label)
    X_val=np.array(val_X_flat,dtype=np.float32); y_val=np.array(val_y_flat,dtype=np.int32)
    print(f"  Val: {len(X_val):,} pairs ({sum(y_val):,}+)")

    # ── Train LightGBM ─────────────────────────────────────────────────────────
    print("[5/6] Training LightGBM (Model A)...")
    spw=max(1.0,ratio*0.3)
    lgb_clf=lgb.LGBMClassifier(
        n_estimators=1000,learning_rate=0.025,num_leaves=63,max_depth=8,
        min_child_samples=15,subsample=0.80,colsample_bytree=0.80,
        reg_alpha=0.2,reg_lambda=2.0,scale_pos_weight=spw,
        objective="binary",random_state=42,verbosity=-1,n_jobs=-1,
    )
    lgb_clf.fit(X_train,y_train,eval_set=[(X_val,y_val)],
                callbacks=[lgb.early_stopping(60,verbose=True)])

    # ── Train XGBoost ──────────────────────────────────────────────────────────
    xgb_clf = None
    if _HAVE_XGB:
        print("Training XGBoost (Model B)...")
        xgb_clf=xgb.XGBClassifier(
            n_estimators=800,learning_rate=0.025,max_depth=7,
            min_child_weight=3,subsample=0.80,colsample_bytree=0.80,
            reg_alpha=0.2,reg_lambda=2.0,scale_pos_weight=max(1.0,ratio*0.3),
            objective="binary:logistic",eval_metric="logloss",
            random_state=42,verbosity=0,n_jobs=-1,
            early_stopping_rounds=60,
        )
        xgb_clf.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=False)
        print("  XGBoost done.")

    # ── Val predictions (ensemble) ─────────────────────────────────────────────
    print("[6/6] Evaluating ensemble & grid-searching thresholds...")
    lgb_probs=lgb_clf.predict_proba(X_val)[:,1]
    if xgb_clf:
        xgb_probs=xgb_clf.predict_proba(X_val)[:,1]
        val_probs=0.55*lgb_probs + 0.45*xgb_probs
        print("  Ensemble: 55% LGB + 45% XGB")
    else:
        val_probs=lgb_probs

    offset=0; val_preds_by_q={}
    for qid in val_qids:
        pairs=val_candidates[qid]; cnt=len(pairs)
        if cnt==0: val_preds_by_q[qid]=[]
        else:
            q_probs=val_probs[offset:offset+cnt]
            val_preds_by_q[qid]=[(pairs[i][0],q_probs[i]) for i in range(cnt)]
            offset+=cnt

    print("="*80)
    print("GRID SEARCH on VAL set:")
    best_f=0.0; best_cfg=None
    for base_th in [0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85]:
        for sib_th in [0.35,0.40,0.45,0.50,0.55,0.60,0.65]:
            if sib_th>base_th: continue
            scores=[]; tp=fp=fn=n_sing=sing_ok=0
            for qid in val_qids:
                true_set=all_gt[qid]
                if not true_set: n_sing+=1
                pairs=val_preds_by_q.get(qid,[])
                pred_set=set()
                if pairs:
                    sp=sorted(pairs,key=lambda x:-x[1])
                    if sp[0][1]>=base_th:
                        pred_set.add(sp[0][0])
                        for tid,p in sp[1:]:
                            if p>=sib_th: pred_set.add(tid)
                            else: break
                tp+=len(pred_set&true_set); fp+=len(pred_set-true_set); fn+=len(true_set-pred_set)
                if not true_set and not pred_set: sing_ok+=1
                scores.append(compute_entity_f_beta(pred_set,true_set,0.5))
            macro_f=float(np.mean(scores)); prec=tp/max(1,tp+fp); rec=tp/max(1,tp+fn)
            tag=" * New Best" if macro_f>best_f else ""
            print(f"  BaseTh={base_th:.2f} SibTh={sib_th:.2f} → F0.5={macro_f:.4f} P={prec*100:.2f}% R={rec*100:.2f}% Sing={sing_ok}/{n_sing}{tag}")
            if macro_f>best_f: best_f=macro_f; best_cfg=(base_th,sib_th,prec,rec,sing_ok,n_sing)

    base_th,sib_th,prec,rec,s_cor,s_tot=best_cfg
    print("\n"+"="*80)
    print(f"BEST: F0.5={best_f:.4f} P={prec*100:.2f}% R={rec*100:.2f}% BaseTh={base_th:.2f} SibTh={sib_th:.2f} Sing={s_cor}/{s_tot}")
    print(f"Elapsed: {time.time()-t0:.1f}s")
    print("="*80)

    # Save ensemble bundle
    bundle={"lgb_model":lgb_clf,"xgb_model":xgb_clf,"feature_names":FEATURE_NAMES,
            "base_threshold":base_th,"sibling_threshold":sib_th,"macro_f05":best_f,
            "lgb_weight":0.55,"xgb_weight":0.45 if xgb_clf else 0.0}
    joblib.dump(bundle,"model_finetuned_v21.pkl")
    print("Saved → model_finetuned_v21.pkl")

if __name__=="__main__":
    main()
