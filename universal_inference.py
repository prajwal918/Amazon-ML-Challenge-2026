#!/usr/bin/env python3
"""
================================================================================
Amazon ML Challenge 2026 -- Universal POWER Inference (V20/V21/V22)
================================================================================
Auto-detects model version and applies correct feature extraction + ensemble.
Saves raw scores + sweeps 13 thresholds + applies hard rules if V22.
Usage:
  python universal_inference.py                    # auto-detect best model
  python universal_inference.py model_finetuned_v21.pkl   # specify model
================================================================================
"""
import sys, os, re, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import joblib
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from unidecode import unidecode as _unidecode

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Metaphone ─────────────────────────────────────────────────────────────────
_HAVE_METAPHONE = False
try:
    import jellyfish; _HAVE_METAPHONE = True
except: pass

def metaphone_sim(a, b):
    if not _HAVE_METAPHONE or not a or not b: return 0.5
    try:
        ma = jellyfish.metaphone(a); mb = jellyfish.metaphone(b)
        return 1.0 if ma == mb else fuzz.ratio(ma, mb) / 100.0
    except: return 0.5

# ── Indic ─────────────────────────────────────────────────────────────────────
_HAVE_INDIC = False
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _translit
    _SCRIPT_RANGES = [(0x0900,0x097F,sanscript.DEVANAGARI),(0x0B80,0x0BFF,sanscript.TAMIL),
                      (0x0980,0x09FF,sanscript.BENGALI),(0x0C00,0x0C7F,sanscript.TELUGU),
                      (0x0C80,0x0CFF,sanscript.KANNADA)]
    _HAVE_INDIC = True
except: pass

def _token_script(t):
    for ch in t:
        c = ord(ch)
        for lo, hi, sc in _SCRIPT_RANGES:
            if lo <= c <= hi: return sc
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

# ── Normalization ──────────────────────────────────────────────────────────────
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

def abbrev_match(q_full,t_full,q_core,t_core):
    qi = "".join(w[0] for w in q_core.split() if w not in _STOPWORDS and len(w)>1)
    ti = "".join(w[0] for w in t_core.split() if w not in _STOPWORDS and len(w)>1)
    if (qi and ti and qi==ti and abs(len(q_core.split())-len(t_core.split()))>=2): return 1.0
    if is_abbreviation(q_core) and ti.startswith(q_core[:len(q_core)]): return 0.9
    if is_abbreviation(t_core) and qi.startswith(t_core[:len(t_core)]): return 0.9
    return 0.0

def normalize_name(raw, version=48):
    """Normalize name. version=35 for V20, 41 for V21, 48 for V22."""
    if not isinstance(raw,str) or not raw.strip():
        d = {"full":"","core":"","comp":"","leet":"","dba_alt":"","tokens":frozenset(),"qgrams":frozenset(),"initials":"","soundex_toks":frozenset()}
        if version >= 41: d["num_tokens"] = frozenset()
        if version >= 48: d["char4grams"] = frozenset(); d["is_short"] = False
        return d
    s = raw.strip(); dba_alt = ""
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
    if version >= 41:
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
    d = {"full":full,"core":core,"comp":comp,"leet":leet,"dba_alt":dba_alt,
         "tokens":tokens,"qgrams":qgrams,"initials":initials,"soundex_toks":soundex_toks}
    if version >= 41:
        d["num_tokens"] = frozenset(t for t in core.split() if t.isdigit())
    if version >= 48:
        d["char4grams"] = frozenset(comp[i:i+4] for i in range(len(comp)-3)) if len(comp)>=4 else frozenset()
        d["is_short"] = len(comp) <= 5
    return d

US_STATES={"al":"alabama","ak":"alaska","az":"arizona","ar":"arkansas","ca":"california","co":"colorado","ct":"connecticut","de":"delaware","fl":"florida","ga":"georgia","hi":"hawaii","id":"idaho","il":"illinois","in":"indiana","ia":"iowa","ks":"kansas","ky":"kentucky","la":"louisiana","me":"maine","md":"maryland","ma":"massachusetts","mi":"michigan","mn":"minnesota","ms":"mississippi","mo":"missouri","mt":"montana","ne":"nebraska","nv":"nevada","nh":"new hampshire","nj":"new jersey","nm":"new mexico","ny":"new york","nc":"north carolina","nd":"north dakota","oh":"ohio","ok":"oklahoma","or":"oregon","pa":"pennsylvania","ri":"rhode island","sc":"south carolina","sd":"south dakota","tn":"tennessee","tx":"texas","ut":"utah","vt":"vermont","va":"virginia","wa":"washington","wv":"west virginia","wi":"wisconsin","wy":"wyoming","dc":"district of columbia"}
IN_STATES={"ap":"andhra pradesh","ar":"arunachal pradesh","as":"assam","br":"bihar","ct":"chhattisgarh","ga":"goa","gj":"gujarat","hr":"haryana","hp":"himachal pradesh","jh":"jharkhand","ka":"karnataka","kl":"kerala","mp":"madhya pradesh","mh":"maharashtra","mn":"manipur","ml":"meghalaya","mz":"mizoram","nl":"nagaland","or":"odisha","pb":"punjab","rj":"rajasthan","sk":"sikkim","tn":"tamil nadu","tg":"telangana","ts":"telangana","tr":"tripura","up":"uttar pradesh","uk":"uttarakhand","wb":"west bengal","dl":"delhi"}
STATE_ABBR={**US_STATES,**IN_STATES}; FULL_TO_ABBR={v:k for k,v in STATE_ABBR.items()}
FULL_NAME_CANON={f:f for f in FULL_TO_ABBR}
FULL_NAME_CANON.update({"orissa":"odisha","uttaranchal":"uttarakhand","pondicherry":"puducherry","tamilnadu":"tamil nadu"})

def normalize_address(raw, country="", version=48):
    if not isinstance(raw,str) or not raw.strip():
        d = {"house_number":"","state":"","postal":"","tokens":frozenset(),"full":"","city":"","is_empty":True}
        if version >= 48: d["tok_count"] = 0
        return d
    s = romanize_mixed(raw); s = _unidecode(s).lower()
    s = _WS_RE.sub(" ",_NONALNUM_RE.sub(" ",s)).strip(); tokens = s.split()
    postal = ""
    if country == "India":
        m = _POSTAL_IN.search(s)
        if m: postal = m.group(0)
    else:
        m = _POSTAL_US.search(s)
        if m: postal = m.group(0)
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
    d = {"house_number":house_number,"state":state,"postal":postal,"tokens":tok_set,"full":s,"city":city,"is_empty":False}
    if version >= 48: d["tok_count"] = len(tokens)
    return d

def compute_features_v20(qn,qa,tn,ta):
    """35 features for V20."""
    qf,tf=qn["full"],tn["full"]; qc,tc=qn["core"],tn["core"]
    tsort=fuzz.token_sort_ratio(qf,tf)/100.0; tset=fuzz.token_set_ratio(qf,tf)/100.0; partial=fuzz.partial_ratio(qf,tf)/100.0
    jw=float(JaroWinkler.similarity(qf,tf))
    core_jw=float(JaroWinkler.similarity(qc,tc)) if qc and tc else 0.0
    core_sort=fuzz.token_sort_ratio(qc,tc)/100.0 if qc and tc else 0.0
    core_set=fuzz.token_set_ratio(qc,tc)/100.0 if qc and tc else 0.0
    exact_core=1.0 if (qc and tc and qc==tc) else 0.0
    exact_comp=1.0 if (qn["comp"] and tn["comp"] and qn["comp"]==tn["comp"]) else 0.0
    ql,tl=len(qf),len(tf); len_ratio=(min(ql,tl)/max(ql,tl)) if max(ql,tl)>0 else 0.0; len_diff=abs(ql-tl)
    itk=len(qn["tokens"]&tn["tokens"]); utk=len(qn["tokens"]|tn["tokens"]); tok_j=itk/utk if utk>0 else 0.0
    iqg=len(qn["qgrams"]&tn["qgrams"]); uqg=len(qn["qgrams"]|tn["qgrams"]); qg_j=iqg/uqg if uqg>0 else 0.0
    used_alt=1.0 if (qn["dba_alt"] or tn["dba_alt"]) else 0.0
    isd=len(qn["soundex_toks"]&tn["soundex_toks"]); usd=len(qn["soundex_toks"]|tn["soundex_toks"]); sd_j=isd/usd if usd>0 else 0.0
    init_m=1.0 if (qn["initials"] and tn["initials"] and qn["initials"]==tn["initials"]) else 0.0
    p_core=fuzz.partial_ratio(qc,tc)/100.0 if qc and tc else 0.0
    leet_m=1.0 if (qn["leet"] and tn["leet"] and qn["leet"]==tn["leet"]) else 0.0
    Qc,Tc=qn["comp"],tn["comp"]
    comp_pfx=1.0 if (Qc and Tc and len(Qc)>=4 and len(Tc)>=4 and Qc[:8]==Tc[:8]) else 0.0
    # addr
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
    return [tsort,tset,partial,jw,core_jw,core_sort,core_set,exact_core,exact_comp,len_ratio,float(len_diff),
            tok_j,float(itk),qg_j,used_alt,sd_j,init_m,p_core,leet_m,comp_pfx,
            hm,hc,sm,sc_,pm,pc,a_sort,a_set,a_jacc,cm,te,ae,nxa,cxa,jxs]

def compute_features_v41(qn,qa,tn,ta):
    """41 features for V21."""
    base = compute_features_v20(qn,qa,tn,ta)  # 35 features
    qc,tc = qn["core"],tn["core"]; qf,tf = qn["full"],tn["full"]
    # Insert 6 new features after index 20 (after comp_prefix, before addr)
    abbr_m = abbrev_match(qf,tf,qc,tc)
    lev_r = float(Levenshtein.normalized_similarity(qf,tf)) if qf and tf else 0.0
    lev_c = float(Levenshtein.normalized_similarity(qc,tc)) if qc and tc else 0.0
    nm = 1.0 if (qn.get("num_tokens") and tn.get("num_tokens") and qn["num_tokens"]==tn["num_tokens"]) else (0.5 if not qn.get("num_tokens") and not tn.get("num_tokens") else 0.0)
    ms = float(metaphone_sim(qc.split()[0] if qc else "",tc.split()[0] if tc else ""))
    Qc,Tc = qn["comp"],tn["comp"]
    clr = (min(len(Qc),len(Tc))/max(len(Qc),len(Tc))) if Qc and Tc and max(len(Qc),len(Tc))>0 else 0.0
    # Build: first 20 from base + abbr + lev_r + lev_c + nm + ms + clr + last 15 from base
    return base[:20]+[abbr_m,lev_r,lev_c,nm,ms,clr]+base[20:]

def compute_features_v48(qn,qa,tn,ta):
    """48 features for V22."""
    v41 = compute_features_v41(qn,qa,tn,ta)  # 41 features
    qc,tc = qn["core"],tn["core"]; Qc,Tc = qn["comp"],tn["comp"]
    # char4gram jaccard
    ic4 = len(qn.get("char4grams",frozenset())&tn.get("char4grams",frozenset()))
    uc4 = len(qn.get("char4grams",frozenset())|tn.get("char4grams",frozenset()))
    c4j = ic4/uc4 if uc4>0 else 0.0
    isb = 1.0 if (qn.get("is_short",False) and tn.get("is_short",False)) else 0.0
    # addr extras
    if ta["is_empty"] or qa["is_empty"]:
        atd=0.0; csc=0.5; spa=0.5
    else:
        atd = min(1.0,abs(qa.get("tok_count",0)-ta.get("tok_count",0))/10.0)
        cm_val = v41[35]  # city_match
        sm_val = v41[28]  # state_match in v41 = index 22 in base + offset... let me compute directly
        qs,ts = qa["state"],ta["state"]
        if qs and ts:
            sm2 = 1.0 if qs==ts else 0.0; sc2 = 0.0 if qs==ts else 1.0
        else: sm2=0.5; sc2=0.0
        qp,tp = qa["postal"],ta["postal"]
        if qp and tp:
            pm2=1.0 if qp==tp else 0.0; pc2=0.0 if qp==tp else 1.0
        else: pm2=0.5; pc2=0.0
        qct,tct = qa["city"],ta["city"]
        cm2 = 1.0 if (qct and tct and qct==tct) else (0.5 if (not qct or not tct) else 0.0)
        csc = 1.0 if (cm2==1.0 and sm2==1.0) else (0.0 if (cm2==0.0 and sm2==0.0) else 0.5)
        if sm2==1.0 and pm2==1.0: spa=1.0
        elif sc2==1.0 or pc2==1.0: spa=0.0
        else: spa=0.5
    # cross
    conf_agree = v41[0]*v41[34]  # tsort * addr_jacc (approx)
    hard_conflict = 1.0 if (v41[29]==1.0 and v41[31]==1.0) else 0.0  # state_conflict AND postal_conflict
    # Insert after v41: c4j, isb (after index 25, before addr), plus atd, csc, spa before cross
    # V22 layout: first 26 from v41 + c4j + isb + addr12 from v41[26:38] + atd + csc + spa + cross5
    return v41[:26]+[c4j,isb]+v41[26:38]+[atd,csc,spa]+v41[38:41]+[conf_agree,hard_conflict]

def hard_rule_reject_v22(feats):
    """Hard rejection for V22 (48 features)."""
    # hard_conflict is last feature (index 47)
    if feats[47] == 1.0: return True
    # house_conflict (index 29) + state_conflict (index 31)
    if feats[29] == 1.0 and feats[31] == 1.0: return True
    return False


def main():
    t0 = time.time()
    print("=" * 80)
    print("UNIVERSAL POWER INFERENCE — AUTO-DETECT MODEL VERSION")
    print("=" * 80)

    # Auto-detect best model
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        for mp in ["model_finetuned_v22.pkl","model_finetuned_v21.pkl","model_finetuned_v20.pkl"]:
            if os.path.exists(mp):
                model_path = mp; break
    if not model_path or not os.path.exists(model_path):
        print("ERROR: No model found! Train a model first."); return

    bundle = joblib.load(model_path)
    n_feats = len(bundle["feature_names"])
    base_th = bundle["base_threshold"]; sib_th = bundle["sibling_threshold"]

    # Detect version
    has_mlp = "mlp_model" in bundle
    has_xgb = bundle.get("xgb_model") is not None
    has_iso = "isotonic" in bundle
    if n_feats == 48: version = "V22"
    elif n_feats == 41: version = "V21"
    else: version = "V20"

    print(f"Model: {model_path} | Version: {version} | Features: {n_feats}")
    print(f"BaseTh={base_th:.2f} SibTh={sib_th:.2f} | offline F0.5={bundle['macro_f05']:.4f}")
    print(f"Ensemble: LGB={'Yes'} XGB={'Yes' if has_xgb else 'No'} MLP={'Yes' if has_mlp else 'No'} Isotonic={'Yes' if has_iso else 'No'}")

    # Extract models
    if "lgb_model" in bundle: lgb_model = bundle["lgb_model"]
    else: lgb_model = bundle["model"]  # V20 compat
    xgb_model = bundle.get("xgb_model")
    mlp_model = bundle.get("mlp_model")
    scaler    = bundle.get("scaler")
    isotonic  = bundle.get("isotonic")
    lgb_w = bundle.get("lgb_weight", 1.0)
    xgb_w = bundle.get("xgb_weight", 0.0)
    mlp_w = bundle.get("mlp_weight", 0.0)

    # Pick feature function
    if n_feats == 48: feat_fn = compute_features_v48; nv = 48
    elif n_feats == 41: feat_fn = compute_features_v41; nv = 41
    else: feat_fn = compute_features_v20; nv = 35

    BASE_DIR = Path("student_resource/dataset/test")
    CAND_FILE = Path("output/candidate_pairs.tsv")
    ver_tag = version.lower()
    RAW_FILE = Path(f"output/raw_scores_{ver_tag}.tsv")
    SWEEP_DIR = Path(f"output/threshold_sweep_{ver_tag}")
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
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
        path = BASE_DIR / fn
        if not path.exists(): continue
        with open(path,"r",encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t"); eid = p[0]
                if eid in needed:
                    tgt_raw[eid] = (p[1], p[2] if len(p)>2 else "", p[3].strip() if len(p)>3 else "")
    print(f"  {len(tgt_raw):,} targets loaded ({time.time()-t0:.1f}s)")

    print("[3/4] Loading queries...")
    s1_norm = {}
    with open(BASE_DIR/"test_source1.tsv","r",encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t"); qid = p[0]
            s1_norm[qid] = (normalize_name(p[1] if len(p)>1 else "", nv),
                            normalize_address(p[2] if len(p)>2 else "", p[3].strip() if len(p)>3 else "", nv))
    print(f"  {len(s1_norm):,} queries")

    # ── Scoring ────────────────────────────────────────────────────────────────
    print(f"[4/4] Batch inference ({version}, {n_feats} features)...")
    score_dict = {}; CHUNK = 50000; n_total = len(s1_order); t_s = time.time()

    for cs in range(0, n_total, CHUNK):
        ce = min(n_total, cs + CHUNK); chunk_qids = s1_order[cs:ce]
        ct = set()
        for qid in chunk_qids:
            for cid in cand_dict[qid]:
                if cid in tgt_raw: ct.add(cid)
        ctn = {}
        for cid in ct:
            nm,ad,cy = tgt_raw[cid]
            ctn[cid] = (normalize_name(nm,nv), normalize_address(ad,cy,nv))
        batch_f = []; batch_c = []; batch_cids = []
        for qid in chunk_qids:
            qn,qa = s1_norm[qid]; cids = cand_dict[qid]; cnt = 0; these = []
            for cid in cids:
                if cid in ctn:
                    tn,ta = ctn[cid]
                    batch_f.append(feat_fn(qn,qa,tn,ta)); these.append(cid); cnt += 1
            batch_c.append(cnt); batch_cids.append(these)
        if batch_f:
            X = np.array(batch_f, dtype=np.float32)
            probs_lgb = lgb_model.predict_proba(X)[:,1]
            combined = probs_lgb * lgb_w
            if xgb_model:
                probs_xgb = xgb_model.predict_proba(X)[:,1]
                combined += probs_xgb * xgb_w
            if mlp_model and scaler:
                if xgb_model:
                    stack_X = np.column_stack([X, probs_lgb, probs_xgb])
                else:
                    stack_X = np.column_stack([X, probs_lgb])
                stack_X_sc = scaler.transform(stack_X)
                probs_mlp = mlp_model.predict_proba(stack_X_sc)[:,1]
                combined += probs_mlp * mlp_w
            if isotonic:
                combined = isotonic.transform(combined)
            offset = 0
            for i, qid in enumerate(chunk_qids):
                cnt = batch_c[i]; cids = batch_cids[i]
                if cnt == 0: score_dict[qid] = []
                else:
                    q_p = combined[offset:offset+cnt]
                    q_f = [batch_f[offset+j] for j in range(cnt)] if version=="V22" else None
                    pairs = sorted(zip(cids, q_p.tolist(), q_f if q_f else [None]*cnt), key=lambda x:-x[1])
                    score_dict[qid] = [(tid,prob,feats) for tid,prob,feats in pairs]
                    offset += cnt
        else:
            for qid in chunk_qids: score_dict[qid] = []
        del ctn, batch_f, batch_cids
        done=ce; el=time.time()-t_s; rate=done/max(1.0,el)
        print(f"  {done:,}/{n_total:,} ({done/n_total*100:.1f}%) | {rate:.0f} q/s | {el:.1f}s")

    print(f"Inference done ({time.time()-t0:.1f}s)")

    # ── Save raw scores ────────────────────────────────────────────────────────
    print(f"Saving raw scores → {RAW_FILE}")
    with open(RAW_FILE,"w",encoding="utf-8") as f:
        f.write("source1_entity_id\tscored_candidates\n")
        for qid in s1_order:
            pairs = score_dict[qid]
            val = ";".join(f"{tid}:{prob:.4f}" for tid,prob,_ in pairs)
            f.write(f"{qid}\t{val}\n")

    # ── Threshold sweep ────────────────────────────────────────────────────────
    print(f"\nTHRESHOLD SWEEP → {SWEEP_DIR}/")
    thresholds = [round(x,2) for x in np.arange(0.30,0.92,0.05)]
    use_hard_rules = version == "V22"

    def write_submission(bth, sth, out_path):
        nm = 0
        with open(out_path,"w",encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for qid in s1_order:
                pairs = score_dict[qid]; chosen = []
                if pairs:
                    top_tid, top_p, top_f = pairs[0]
                    if top_p >= bth:
                        reject = use_hard_rules and top_f and hard_rule_reject_v22(top_f)
                        if not reject:
                            chosen.append(top_tid)
                            for tid,p,feat in pairs[1:]:
                                if p >= sth:
                                    if use_hard_rules and feat and hard_rule_reject_v22(feat): continue
                                    chosen.append(tid)
                                else: break
                if chosen: nm += 1
                f.write(f"{qid}\t{','.join(chosen)}\n")
        pct = nm/max(1,len(s1_order))*100
        print(f"  th={bth:.2f}/{sth:.2f} → {nm:,} matched ({pct:.1f}%)")
        return nm

    for th in thresholds:
        sth = max(0.25, th - 0.05)
        write_submission(th, sth, SWEEP_DIR / f"submission_th{int(th*100):02d}.tsv")

    # Write final: model's optimal threshold
    FINAL = Path("output/matching_results.tsv")
    FINAL_VER = Path(f"output/matching_results_{ver_tag}.tsv")
    print(f"\nFinal submission (BaseTh={base_th:.2f}, SibTh={sib_th:.2f}):")
    write_submission(base_th, sib_th, FINAL)
    write_submission(base_th, sib_th, FINAL_VER)

    ns = sum(1 for v in score_dict.values() if not any(p >= base_th for _,p,_ in v)) if score_dict else 0
    print(f"\nTotal: {len(s1_order):,} | Runtime: {time.time()-t0:.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
