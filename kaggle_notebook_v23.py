#!/usr/bin/env python
# coding: utf-8
"""
Amazon ML Challenge 2026 — V23 Kaggle Power Notebook
LightGBM + XGBoost Ensemble, 41 features, BaseTh=0.85
Runs on Kaggle with GPU acceleration.
Outputs: matching_results.tsv + raw_scores.tsv
"""

import os, re, time, random, sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rapidfuzz", "unidecode", "indic-transliteration", "jellyfish"], check=False)

from collections import defaultdict
from pathlib import Path
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from unidecode import unidecode as _unidecode
import lightgbm as lgb
import xgboost as xgb
import joblib

try:
    import jellyfish; _HAVE_META = True
except: _HAVE_META = False

def metaphone_sim(a, b):
    if not _HAVE_META or not a or not b: return 0.5
    try:
        ma = jellyfish.metaphone(a); mb = jellyfish.metaphone(b)
        return 1.0 if ma == mb else fuzz.ratio(ma, mb) / 100.0
    except: return 0.5

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _translit
    _SCRIPT_RANGES = [(0x0900,0x097F,sanscript.DEVANAGARI),(0x0B80,0x0BFF,sanscript.TAMIL),(0x0980,0x09FF,sanscript.BENGALI),(0x0C00,0x0C7F,sanscript.TELUGU),(0x0C80,0x0CFF,sanscript.KANNADA)]
    _HAVE_INDIC = True
except: _HAVE_INDIC = False

def _token_script(t):
    for ch in t:
        c = ord(ch)
        for lo,hi,sc in _SCRIPT_RANGES:
            if lo<=c<=hi: return sc
    return None

def romanize_mixed(text):
    if not text or not _HAVE_INDIC: return text
    out = []
    for tok in text.split():
        sc = _token_script(tok)
        if sc:
            try: out.append(_translit(tok, sc, sanscript.ITRANS))
            except: out.append(tok)
        else: out.append(tok)
    return " ".join(out)

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

LEGAL_SUFFIXES = ["private limited","pvt ltd","pvt. ltd.","pvt","llc","l.l.c","incorporated","inc","corp","corporation","services","center","centre","holdings","group","enterprises","solutions","llp","l.l.p","sarl","sas","ltd","limited","co","company","pllc","plc","gmbh","sasu","eurl","sa","praiveta limiteda","praivett limitted","industries","associates","consulting","international","intl"]
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
def is_abbreviation(s):
    words = s.upper().split()
    return len(words)==1 and bool(_ABBR_RE.match(words[0])) if words else False
def abbrev_match(qf,tf,qc,tc):
    qi = "".join(w[0] for w in qc.split() if w not in _STOPWORDS and len(w)>1)
    ti = "".join(w[0] for w in tc.split() if w not in _STOPWORDS and len(w)>1)
    if qi and ti and qi==ti and abs(len(qc.split())-len(tc.split()))>=2: return 1.0
    if is_abbreviation(qc) and ti.startswith(qc[:len(qc)]): return 0.9
    if is_abbreviation(tc) and qi.startswith(tc[:len(tc)]): return 0.9
    return 0.0

def normalize_name(raw):
    if not isinstance(raw,str) or not raw.strip():
        return {"full":"","core":"","comp":"","leet":"","dba_alt":"","tokens":frozenset(),"qgrams":frozenset(),"initials":"","soundex_toks":frozenset(),"num_tokens":frozenset()}
    s = raw.strip(); dba_alt = ""
    m = _DBA_RE.search(s)
    if m:
        after = s[m.end():].strip()
        if after: dba_alt = _clean_basic(_unidecode(after).lower()); dba_alt = _WS_RE.sub(" ",_SUFFIX_RE.sub(" ",dba_alt)).strip()
        s = s[:m.start()].strip()
    s = _METADATA_RE.sub(" ",s); s = _HANDLE_RE.sub(" ",s); s = _DOMAIN_RE.sub(" ",s)
    s = romanize_mixed(s); s = _unidecode(s).lower()
    for word,digit in _NUM_WORD.items(): s = re.sub(r'\b'+word+r'\b',digit,s)
    for word,digit in _ORDINAL.items(): s = re.sub(r'\b'+word+r'\b',digit,s)
    full = _clean_basic(s); core = _WS_RE.sub(" ",_SUFFIX_RE.sub(" ",full)).strip()
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
    return {"full":full,"core":core,"comp":comp,"leet":leet,"dba_alt":dba_alt,"tokens":tokens,"qgrams":qgrams,"initials":initials,"soundex_toks":soundex_toks,"num_tokens":num_tokens}

US_STATES={"al":"alabama","ak":"alaska","az":"arizona","ar":"arkansas","ca":"california","co":"colorado","ct":"connecticut","de":"delaware","fl":"florida","ga":"georgia","hi":"hawaii","id":"idaho","il":"illinois","in":"indiana","ia":"iowa","ks":"kansas","ky":"kentucky","la":"louisiana","me":"maine","md":"maryland","ma":"massachusetts","mi":"michigan","mn":"minnesota","ms":"mississippi","mo":"missouri","mt":"montana","ne":"nebraska","nv":"nevada","nh":"new hampshire","nj":"new jersey","nm":"new mexico","ny":"new york","nc":"north carolina","nd":"north dakota","oh":"ohio","ok":"oklahoma","or":"oregon","pa":"pennsylvania","ri":"rhode island","sc":"south carolina","sd":"south dakota","tn":"tennessee","tx":"texas","ut":"utah","vt":"vermont","va":"virginia","wa":"washington","wv":"west virginia","wi":"wisconsin","wy":"wyoming","dc":"district of columbia"}
IN_STATES={"ap":"andhra pradesh","ar":"arunachal pradesh","as":"assam","br":"bihar","ct":"chhattisgarh","ga":"goa","gj":"gujarat","hr":"haryana","hp":"himachal pradesh","jh":"jharkhand","ka":"karnataka","kl":"kerala","mp":"madhya pradesh","mh":"maharashtra","mn":"manipur","ml":"meghalaya","mz":"mizoram","nl":"nagaland","or":"odisha","pb":"punjab","rj":"rajasthan","sk":"sikkim","tn":"tamil nadu","tg":"telangana","ts":"telangana","tr":"tripura","up":"uttar pradesh","uk":"uttarakhand","wb":"west bengal","dl":"delhi"}
STATE_ABBR={**US_STATES,**IN_STATES}; FULL_TO_ABBR={v:k for k,v in STATE_ABBR.items()}
FULL_NAME_CANON={f:f for f in FULL_TO_ABBR}
FULL_NAME_CANON.update({"orissa":"odisha","uttaranchal":"uttarakhand","pondicherry":"puducherry","tamilnadu":"tamil nadu"})

def normalize_address(raw,country=""):
    if not isinstance(raw,str) or not raw.strip():
        return {"house_number":"","state":"","postal":"","tokens":frozenset(),"full":"","city":"","is_empty":True}
    s = romanize_mixed(raw); s = _unidecode(s).lower(); s = _WS_RE.sub(" ",_NONALNUM_RE.sub(" ",s)).strip(); tokens = s.split()
    postal = ""
    if country=="India":
        m = _POSTAL_IN.search(s)
        if m: postal = m.group(0)
    else:
        m = _POSTAL_US.search(s)
        if m: postal = m.group(0)
    house_number = ""
    for tok in tokens:
        if tok.isdigit() and tok!=postal and len(tok)<=5: house_number = tok.lstrip("0") or "0"; break
    if not house_number:
        for tok in tokens:
            mm = _NUM_RE.search(tok)
            if mm:
                val = mm.group(0)
                if val!=postal and len(val)<=5: house_number = val.lstrip("0") or "0"; break
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
        if len(tok)>=3 and tok not in _STOPWORDS and tok not in STATE_ABBR and not tok.isdigit() and tok!=postal: city = tok; break
    tok_set = frozenset(t for t in tokens if t not in _STOPWORDS and len(t)>1)
    return {"house_number":house_number,"state":state,"postal":postal,"tokens":tok_set,"full":s,"city":city,"is_empty":False}

FEATURE_NAMES = ["name_token_sort","name_token_set","name_partial","name_jw","name_core_jw","name_core_sort","name_core_set","name_exact_core","name_exact_comp","name_len_ratio","name_len_diff","name_token_jaccard","name_tok_overlap_cnt","name_qgram3_jaccard","used_alt_name","name_soundex_jaccard","name_initials_match","name_partial_core","name_leet_match","name_comp_prefix_match","name_abbreviation_match","name_levenshtein_ratio","name_levenshtein_core","name_num_tok_match","name_metaphone_sim","name_comp_len_ratio","addr_house_match","addr_house_conflict","addr_state_match","addr_state_conflict","addr_postal_match","addr_postal_conflict","addr_token_sort","addr_token_set","addr_token_jaccard","addr_city_match","addr_target_empty","addr_exact_full","name_x_addr","core_x_addr_set","name_jw_x_addr_state"]

def compute_pair_features(qn,qa,tn,ta):
    qf,tf=qn["full"],tn["full"]; qc,tc=qn["core"],tn["core"]
    tsort=fuzz.token_sort_ratio(qf,tf)/100.0; tset=fuzz.token_set_ratio(qf,tf)/100.0; partial=fuzz.partial_ratio(qf,tf)/100.0
    jw=float(JaroWinkler.similarity(qf,tf)); core_jw=float(JaroWinkler.similarity(qc,tc)) if qc and tc else 0.0
    core_sort=fuzz.token_sort_ratio(qc,tc)/100.0 if qc and tc else 0.0; core_set=fuzz.token_set_ratio(qc,tc)/100.0 if qc and tc else 0.0
    exact_core=1.0 if (qc and tc and qc==tc) else 0.0; exact_comp=1.0 if (qn["comp"] and tn["comp"] and qn["comp"]==tn["comp"]) else 0.0
    ql,tl=len(qf),len(tf); len_ratio=(min(ql,tl)/max(ql,tl)) if max(ql,tl)>0 else 0.0; len_diff=abs(ql-tl)
    itk=len(qn["tokens"]&tn["tokens"]); utk=len(qn["tokens"]|tn["tokens"]); tok_j=itk/utk if utk>0 else 0.0
    iqg=len(qn["qgrams"]&tn["qgrams"]); uqg=len(qn["qgrams"]|tn["qgrams"]); qg_j=iqg/uqg if uqg>0 else 0.0
    used_alt=1.0 if (qn["dba_alt"] or tn["dba_alt"]) else 0.0
    isd=len(qn["soundex_toks"]&tn["soundex_toks"]); usd=len(qn["soundex_toks"]|tn["soundex_toks"]); sd_j=isd/usd if usd>0 else 0.0
    init_m=1.0 if (qn["initials"] and tn["initials"] and qn["initials"]==tn["initials"]) else 0.0
    p_core=fuzz.partial_ratio(qc,tc)/100.0 if qc and tc else 0.0
    leet_m=1.0 if (qn["leet"] and tn["leet"] and qn["leet"]==tn["leet"]) else 0.0
    Qc,Tc=qn["comp"],tn["comp"]; comp_pfx=1.0 if (Qc and Tc and len(Qc)>=4 and len(Tc)>=4 and Qc[:8]==Tc[:8]) else 0.0
    abbr_m=abbrev_match(qf,tf,qc,tc)
    lev_r=float(Levenshtein.normalized_similarity(qf,tf)) if qf and tf else 0.0
    lev_c=float(Levenshtein.normalized_similarity(qc,tc)) if qc and tc else 0.0
    nm=1.0 if (qn["num_tokens"] and tn["num_tokens"] and qn["num_tokens"]==tn["num_tokens"]) else (0.5 if not qn["num_tokens"] and not tn["num_tokens"] else 0.0)
    ms=float(metaphone_sim(qc.split()[0] if qc else "",tc.split()[0] if tc else ""))
    clr=(min(len(Qc),len(Tc))/max(len(Qc),len(Tc))) if Qc and Tc and max(len(Qc),len(Tc))>0 else 0.0
    if ta["is_empty"] or qa["is_empty"]:
        hm=hc=sm=sc_=pm=pc=a_sort=a_set=a_jacc=0.5; te=1.0 if ta["is_empty"] else 0.0; cm=0.5; ae=0.0
    else:
        te=0.0; qh,th=qa["house_number"],ta["house_number"]
        if qh and th:
            if qh==th: hm,hc=1.0,0.0
            elif qh.startswith(th) or th.startswith(qh): hm,hc=0.8,0.0
            else: hm,hc=0.0,1.0
        else: hm,hc=0.5,0.0
        qs,ts=qa["state"],ta["state"]
        if qs and ts:
            if qs==ts: sm,sc_=1.0,0.0
            else: sm,sc_=0.0,1.0
        else: sm,sc_=0.5,0.0
        qp,tp=qa["postal"],ta["postal"]
        if qp and tp:
            if qp==tp: pm,pc=1.0,0.0
            else: pm,pc=0.0,1.0
        else: pm,pc=0.5,0.0
        a_sort=fuzz.token_sort_ratio(qa["full"],ta["full"])/100.0; a_set=fuzz.token_set_ratio(qa["full"],ta["full"])/100.0
        ai=len(qa["tokens"]&ta["tokens"]); au=len(qa["tokens"]|ta["tokens"]); a_jacc=ai/au if au>0 else 0.0
        qct,tct=qa["city"],ta["city"]; cm=1.0 if (qct and tct and qct==tct) else (0.5 if (not qct or not tct) else 0.0)
        ae=1.0 if (qa["full"] and ta["full"] and qa["full"]==ta["full"]) else 0.0
    nxa=tsort*a_sort; cxa=core_set*a_set; jxs=jw*sm
    return [tsort,tset,partial,jw,core_jw,core_sort,core_set,exact_core,exact_comp,len_ratio,float(len_diff),tok_j,float(itk),qg_j,used_alt,sd_j,init_m,p_core,leet_m,comp_pfx,abbr_m,float(lev_r),float(lev_c),nm,float(ms),clr,hm,hc,sm,sc_,pm,pc,a_sort,a_set,a_jacc,cm,te,ae,nxa,cxa,jxs]

def compute_entity_f_beta(pred,true,beta=0.5):
    if not pred and not true: return 1.0
    if not pred or not true: return 0.0
    tp=len(pred&true)
    if tp==0: return 0.0
    prec=tp/len(pred); rec=tp/len(true); b2=beta*beta
    return (1+b2)*prec*rec/(b2*prec+rec)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
print("="*80)
print("V23 KAGGLE POWER — 41-feature LGB+XGB Ensemble")
print("="*80)

# Detect dataset path (Kaggle vs local)
KAGGLE_INPUT = Path("/kaggle/input/amazon-ml-challenge-2026")
LOCAL_TRAIN = Path("student_resource/dataset/train")
LOCAL_TEST = Path("student_resource/dataset/test")

if KAGGLE_INPUT.exists():
    TRAIN_DIR = KAGGLE_INPUT
    TEST_DIR = KAGGLE_INPUT
    CAND_FILE = KAGGLE_INPUT / "candidate_pairs.tsv"
    OUT_DIR = Path("/kaggle/working")
    print("Running on KAGGLE")
else:
    TRAIN_DIR = LOCAL_TRAIN
    TEST_DIR = LOCAL_TEST
    CAND_FILE = Path("output/candidate_pairs.tsv")
    OUT_DIR = Path("output")
    print("Running LOCALLY")

t0 = time.time()

# ═══ PHASE 1: TRAINING ═══
print("\n" + "="*80)
print("PHASE 1: TRAINING")
print("="*80)

print("[1/6] Loading GT...")
all_gt = defaultdict(set)
with open(TRAIN_DIR/"train_ground_truth.tsv","r",encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if len(p)>=2 and p[1].strip():
            for t in p[1].strip().split(","):
                t=t.strip()
                if t: all_gt[p[0].strip()].add(t)
s1_all = set()
with open(TRAIN_DIR/"train_source1.tsv","r",encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if p: s1_all.add(p[0].strip())
for sid in s1_all:
    if sid not in all_gt: all_gt[sid] = set()
needed_tgts = set()
for v in all_gt.values(): needed_tgts |= v
print(f"  {len(all_gt):,} queries | {len(needed_tgts):,} targets")

# Stratified 40K sample
MAX_Q = 40000
matched_keys = [k for k,v in all_gt.items() if v]
singleton_keys = [k for k,v in all_gt.items() if not v]
random.seed(42)
n_m = min(int(MAX_Q*0.65),len(matched_keys)); n_s = min(MAX_Q-n_m,len(singleton_keys))
sampled = random.sample(matched_keys,n_m) + random.sample(singleton_keys,n_s)
all_gt = {k:all_gt[k] for k in sampled}
needed_tgts = set()
for v in all_gt.values(): needed_tgts |= v
print(f"  Sampled: {n_m:,}+{n_s:,}={len(all_gt):,}")

print("[2/6] Loading records...")
s1_records = {}
with open(TRAIN_DIR/"train_source1.tsv","r",encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if p[0] in all_gt: s1_records[p[0]] = (p[1] if len(p)>1 else "",p[2] if len(p)>2 else "",p[3].strip() if len(p)>3 else "")
target_records = {}; dist_loaded = 0; MAX_DIST = 130000
for fn in ["train_source2.tsv","train_source3.tsv"]:
    with open(TRAIN_DIR/fn,"r",encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t"); eid = p[0]
            if eid in needed_tgts: target_records[eid] = (p[1],p[2] if len(p)>2 else "",p[3].strip() if len(p)>3 else "")
            elif dist_loaded < MAX_DIST: target_records[eid] = (p[1],p[2] if len(p)>2 else "",p[3].strip() if len(p)>3 else ""); dist_loaded += 1
print(f"  {len(target_records):,} targets")

print("[3/6] Normalizing...")
s1_norm = {sid:(normalize_name(n),normalize_address(a,c),c) for sid,(n,a,c) in s1_records.items()}
tgt_norm = {tid:(normalize_name(n),normalize_address(a,c),c) for tid,(n,a,c) in target_records.items()}
comp_idx = defaultdict(list); token_idx = defaultdict(list); hn_idx = defaultdict(list)
for tid,(tn,ta,ctry) in tgt_norm.items():
    if tn["comp"] and len(tn["comp"])>=3: comp_idx[(ctry,tn["comp"][:6])].append(tid)
    for tok in tn["tokens"]:
        if len(tok)>=3: token_idx[(ctry,tok)].append(tid)
    if ta["house_number"]: hn_idx[(ctry,ta["house_number"])].append(tid)

print("[4/6] Generating pairs...")
qids = list(s1_records.keys()); random.shuffle(qids)
val_split = int(len(qids)*0.20); val_qids = set(qids[:val_split]); train_qids = set(qids[val_split:])
train_X,train_y = [],[]; val_candidates = defaultdict(list); n_pos=n_neg=0
for qid in qids:
    qn,qa,q_ctry = s1_norm[qid]; true_set = all_gt[qid]; is_val = qid in val_qids
    cands = set(true_set)
    if qn["comp"] and len(qn["comp"])>=3: cands.update(comp_idx.get((q_ctry,qn["comp"][:6]),[])[:6])
    for tok in qn["tokens"]:
        cands.update(token_idx.get((q_ctry,tok),[])[:4])
        if len(cands)>=25: break
    if qa["house_number"]: cands.update(hn_idx.get((q_ctry,qa["house_number"]),[])[:3])
    cands = [tid for tid in cands if tid in tgt_norm]
    for tid in cands:
        tn,ta,_ = tgt_norm[tid]; feats = compute_pair_features(qn,qa,tn,ta); label = 1 if tid in true_set else 0
        if is_val: val_candidates[qid].append((tid,feats,label))
        else: train_X.append(feats); train_y.append(label); n_pos += label==1; n_neg += label==0
ratio = n_neg/max(1,n_pos); print(f"  {len(train_X):,} pairs | +{n_pos:,} -{n_neg:,}")
X_train = np.array(train_X,dtype=np.float32); y_train = np.array(train_y,dtype=np.int32)
val_Xf,val_yf = [],[]
for qid in val_qids:
    for tid,feats,label in val_candidates[qid]: val_Xf.append(feats); val_yf.append(label)
X_val = np.array(val_Xf,dtype=np.float32); y_val = np.array(val_yf,dtype=np.int32)

print("[5/6] Training LGB+XGB ensemble...")
spw = max(1.0,ratio*0.3)
lgb_clf = lgb.LGBMClassifier(n_estimators=1000,learning_rate=0.025,num_leaves=63,max_depth=8,min_child_samples=15,subsample=0.80,colsample_bytree=0.80,reg_alpha=0.2,reg_lambda=2.0,scale_pos_weight=spw,objective="binary",random_state=42,verbosity=-1,n_jobs=-1)
lgb_clf.fit(X_train,y_train,eval_set=[(X_val,y_val)],callbacks=[lgb.early_stopping(60,verbose=True)])
xgb_clf = xgb.XGBClassifier(n_estimators=800,learning_rate=0.025,max_depth=7,min_child_weight=3,subsample=0.80,colsample_bytree=0.80,reg_alpha=0.2,reg_lambda=2.0,scale_pos_weight=spw,objective="binary:logistic",eval_metric="logloss",random_state=42,verbosity=0,n_jobs=-1,early_stopping_rounds=60)
xgb_clf.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=False)
print("  Models trained.")

# Grid search
lgb_val = lgb_clf.predict_proba(X_val)[:,1]; xgb_val = xgb_clf.predict_proba(X_val)[:,1]
val_probs = 0.55*lgb_val + 0.45*xgb_val
offset = 0; val_preds_by_q = {}
for qid in val_qids:
    pairs = val_candidates[qid]; cnt = len(pairs)
    if cnt==0: val_preds_by_q[qid] = []
    else: q_p = val_probs[offset:offset+cnt]; val_preds_by_q[qid] = [(pairs[i][0],q_p[i]) for i in range(cnt)]; offset += cnt

print("[6/6] Grid search...")
best_f=0.0; best_cfg=None
for base_th in [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]:
    for sib_th in [0.40,0.45,0.50,0.55,0.60,0.65]:
        if sib_th>base_th: continue
        scores=[]
        for qid in val_qids:
            true_set=all_gt[qid]; pairs=val_preds_by_q.get(qid,[]); pred_set=set()
            if pairs:
                sp=sorted(pairs,key=lambda x:-x[1])
                if sp[0][1]>=base_th:
                    pred_set.add(sp[0][0])
                    for tid,p in sp[1:]:
                        if p>=sib_th: pred_set.add(tid)
                        else: break
            scores.append(compute_entity_f_beta(pred_set,true_set,0.5))
        macro_f=float(np.mean(scores))
        if macro_f>best_f: best_f=macro_f; best_cfg=(base_th,sib_th); print(f"  * BaseTh={base_th:.2f} SibTh={sib_th:.2f} → F0.5={macro_f:.4f}")
base_th,sib_th = best_cfg
print(f"\nBest: F0.5={best_f:.4f} BaseTh={base_th:.2f} SibTh={sib_th:.2f}")
print(f"Training time: {time.time()-t0:.1f}s")

# ═══ PHASE 2: INFERENCE ═══
print("\n" + "="*80)
print("PHASE 2: BATCH INFERENCE")
print("="*80)
del train_X, train_y, X_train, y_train, val_Xf, val_yf, X_val, y_val
del s1_norm, tgt_norm, comp_idx, token_idx, hn_idx

t1 = time.time()
print("[1/4] Reading candidates...")
cand_dict = {}; needed = set(); s1_order = []
with open(CAND_FILE,"r",encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t"); qid = p[0]; s1_order.append(qid)
        cids = [x.strip() for x in p[1].split(",") if x.strip()] if len(p)>1 and p[1] else []
        cand_dict[qid] = cids; needed.update(cids)
print(f"  {len(s1_order):,} queries | {len(needed):,} targets")

print("[2/4] Loading targets...")
tgt_raw = {}
for fn in ["test_source2.tsv","test_source3.tsv"]:
    path = TEST_DIR/fn
    if not path.exists(): continue
    with open(path,"r",encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t"); eid = p[0]
            if eid in needed: tgt_raw[eid] = (p[1],p[2] if len(p)>2 else "",p[3].strip() if len(p)>3 else "")
print(f"  {len(tgt_raw):,} targets")

print("[3/4] Loading queries...")
s1_norm = {}
with open(TEST_DIR/"test_source1.tsv","r",encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t"); qid = p[0]
        s1_norm[qid] = (normalize_name(p[1] if len(p)>1 else ""),normalize_address(p[2] if len(p)>2 else "",p[3].strip() if len(p)>3 else ""))
print(f"  {len(s1_norm):,} queries")

print("[4/4] Scoring...")
match_dict = {}; CHUNK = 50000; n_total = len(s1_order); t_s = time.time()
for cs in range(0,n_total,CHUNK):
    ce = min(n_total,cs+CHUNK); chunk_qids = s1_order[cs:ce]
    ct = set()
    for qid in chunk_qids:
        for cid in cand_dict[qid]:
            if cid in tgt_raw: ct.add(cid)
    ctn = {}
    for cid in ct:
        nm,ad,cy = tgt_raw[cid]; ctn[cid] = (normalize_name(nm),normalize_address(ad,cy))
    bf=[]; bc=[]; bcids=[]
    for qid in chunk_qids:
        qn,qa = s1_norm[qid]; cids=cand_dict[qid]; cnt=0; these=[]
        for cid in cids:
            if cid in ctn:
                tn,ta=ctn[cid]; bf.append(compute_pair_features(qn,qa,tn,ta)); these.append(cid); cnt+=1
        bc.append(cnt); bcids.append(these)
    if bf:
        X=np.array(bf,dtype=np.float32)
        probs=0.55*lgb_clf.predict_proba(X)[:,1]+0.45*xgb_clf.predict_proba(X)[:,1]
        offset=0
        for i,qid in enumerate(chunk_qids):
            cnt=bc[i]; cids=bcids[i]
            if cnt==0: match_dict[qid]=""
            else:
                q_p=probs[offset:offset+cnt]; si=np.argsort(q_p)[::-1]; chosen=[]
                if q_p[si[0]]>=base_th:
                    chosen.append(cids[si[0]])
                    for j in si[1:]:
                        if q_p[j]>=sib_th: chosen.append(cids[j])
                        else: break
                match_dict[qid]=",".join(chosen); offset+=cnt
    else:
        for qid in chunk_qids: match_dict[qid]=""
    del ctn,bf,bcids
    done=ce; el=time.time()-t_s; rate=done/max(1.0,el)
    print(f"  {done:,}/{n_total:,} ({done/n_total*100:.1f}%) | {rate:.0f} q/s | {el:.1f}s")

# Write output
out_file = OUT_DIR/"matching_results.tsv"
with open(out_file,"w",encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for qid in s1_order: f.write(f"{qid}\t{match_dict.get(qid,'')}\n")
ns = sum(1 for m in match_dict.values() if not m); nm = len(match_dict)-ns
print(f"\nResults: {len(match_dict):,} total | {nm:,} matched ({nm/len(match_dict)*100:.1f}%) | {ns:,} singletons")
print(f"Total runtime: {time.time()-t0:.1f}s")
print(f"Output: {out_file}")
