#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 -- V20 POWER Inference
 - Saves raw per-pair scores to output/raw_scores_v20.tsv
 - Sweeps thresholds 0.30 → 0.90 and writes a submission for each
 - Best threshold file → output/matching_results.tsv
Usage: python apply_model_v20_power.py
"""
import sys, os, re, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import joblib
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from unidecode import unidecode as _unidecode

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

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
except Exception:
    pass

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

LEGAL_SUFFIXES = [
    "private limited","pvt ltd","pvt. ltd.","pvt","llc","l.l.c",
    "incorporated","inc","corp","corporation","services","center","centre",
    "holdings","group","enterprises","solutions","llp","l.l.p","sarl","sas",
    "ltd","limited","co","company","pllc","plc","gmbh","sasu","eurl","sa",
    "praiveta limiteda","praivett limitted","industries","associates",
    "consulting","international","intl"
]
_SUFFIX_RE   = re.compile(r"\b("+"|".join(sorted((re.escape(s) for s in LEGAL_SUFFIXES),key=len,reverse=True))+r")\.?\b",re.IGNORECASE)
_METADATA_RE = re.compile(r"\(\s*id\s*[:#]?\s*\d+\s*\)",re.IGNORECASE)
_HANDLE_RE   = re.compile(r"(?<![a-z0-9])@")
_DOMAIN_RE   = re.compile(r"\.(com|net|org|io|co|in|fr|biz|org\.in|gov\.in)\b",re.IGNORECASE)
_DBA_RE      = re.compile(r"\bd\.?\s*b\.?\s*a\.?\s*:?\s*",re.IGNORECASE)
_LEET_MAP    = str.maketrans({"5":"s","3":"e","1":"i","0":"o","4":"a","7":"t","8":"b"})
_NONALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE       = re.compile(r"\s+")
_STOPWORDS   = {"the","and","of","a","an","in","to","for","at","on","by","null","near","opp","no"}
_NUM_WORD    = {"one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}

def _clean_basic(s): return _WS_RE.sub(" ",_NONALNUM_RE.sub(" ",s)).strip()

def normalize_name(raw):
    if not isinstance(raw,str) or not raw.strip():
        return {"full":"","core":"","comp":"","leet":"","dba_alt":"","tokens":frozenset(),"qgrams":frozenset(),"initials":"","soundex_toks":frozenset()}
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
    return {"full":full,"core":core,"comp":comp,"leet":leet,"dba_alt":dba_alt,"tokens":tokens,"qgrams":qgrams,"initials":initials,"soundex_toks":soundex_toks}

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
        m=_POSTAL_IN.search(s)
        if m: postal=m.group(0)
    else:
        m=_POSTAL_US.search(s)
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

def compute_pair_features(q_norm,q_addr,t_norm,t_addr):
    q_full,t_full=q_norm["full"],t_norm["full"]; q_core,t_core=q_norm["core"],t_norm["core"]
    tsort=fuzz.token_sort_ratio(q_full,t_full)/100.0; tset=fuzz.token_set_ratio(q_full,t_full)/100.0; partial=fuzz.partial_ratio(q_full,t_full)/100.0
    jw=float(JaroWinkler.similarity(q_full,t_full))
    core_jw=float(JaroWinkler.similarity(q_core,t_core)) if q_core and t_core else 0.0
    core_sort=fuzz.token_sort_ratio(q_core,t_core)/100.0 if q_core and t_core else 0.0
    core_set=fuzz.token_set_ratio(q_core,t_core)/100.0 if q_core and t_core else 0.0
    exact_core=1.0 if (q_core and t_core and q_core==t_core) else 0.0
    exact_comp=1.0 if (q_norm["comp"] and t_norm["comp"] and q_norm["comp"]==t_norm["comp"]) else 0.0
    qlen,tlen=len(q_full),len(t_full); len_ratio=(min(qlen,tlen)/max(qlen,tlen)) if max(qlen,tlen)>0 else 0.0; len_diff=abs(qlen-tlen)
    inter_tok=len(q_norm["tokens"]&t_norm["tokens"]); union_tok=len(q_norm["tokens"]|t_norm["tokens"]); tok_jaccard=inter_tok/union_tok if union_tok>0 else 0.0
    inter_qg=len(q_norm["qgrams"]&t_norm["qgrams"]); union_qg=len(q_norm["qgrams"]|t_norm["qgrams"]); qgram_jacc=inter_qg/union_qg if union_qg>0 else 0.0
    used_alt=1.0 if (q_norm["dba_alt"] or t_norm["dba_alt"]) else 0.0
    inter_sd=len(q_norm["soundex_toks"]&t_norm["soundex_toks"]); union_sd=len(q_norm["soundex_toks"]|t_norm["soundex_toks"]); soundex_jacc=inter_sd/union_sd if union_sd>0 else 0.0
    initials_match=1.0 if (q_norm["initials"] and t_norm["initials"] and q_norm["initials"]==t_norm["initials"]) else 0.0
    partial_core=fuzz.partial_ratio(q_core,t_core)/100.0 if q_core and t_core else 0.0
    leet_match=1.0 if (q_norm["leet"] and t_norm["leet"] and q_norm["leet"]==t_norm["leet"]) else 0.0
    qc,tc=q_norm["comp"],t_norm["comp"]; comp_prefix=1.0 if (qc and tc and len(qc)>=4 and len(tc)>=4 and qc[:8]==tc[:8]) else 0.0
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
        addr_sort=fuzz.token_sort_ratio(q_addr["full"],t_addr["full"])/100.0; addr_set=fuzz.token_set_ratio(q_addr["full"],t_addr["full"])/100.0
        ainter=len(q_addr["tokens"]&t_addr["tokens"]); aunion=len(q_addr["tokens"]|t_addr["tokens"]); addr_jacc=ainter/aunion if aunion>0 else 0.0
        qct,tct=q_addr["city"],t_addr["city"]; city_match=1.0 if (qct and tct and qct==tct) else (0.5 if (not qct or not tct) else 0.0)
        addr_exact=1.0 if (q_addr["full"] and t_addr["full"] and q_addr["full"]==t_addr["full"]) else 0.0
    name_x_addr=tsort*addr_sort; core_x_addr=core_set*addr_set; jw_x_state=jw*state_match
    return [tsort,tset,partial,jw,core_jw,core_sort,core_set,exact_core,exact_comp,len_ratio,float(len_diff),tok_jaccard,float(inter_tok),qgram_jacc,used_alt,soundex_jacc,initials_match,partial_core,leet_match,comp_prefix,house_match,house_conflict,state_match,state_conflict,postal_match,postal_conflict,addr_sort,addr_set,addr_jacc,city_match,target_empty,addr_exact,name_x_addr,core_x_addr,jw_x_state]


def apply_threshold(score_dict, s1_order, base_th, sib_th, out_path):
    """Write a submission file using the given thresholds on saved scores."""
    n_matched = 0
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for qid in s1_order:
            pairs = score_dict.get(qid, [])  # list of (tid, prob) sorted desc
            chosen = []
            if pairs and pairs[0][1] >= base_th:
                chosen.append(pairs[0][0])
                for tid, prob in pairs[1:]:
                    if prob >= sib_th: chosen.append(tid)
                    else: break
            if chosen: n_matched += 1
            f.write(f"{qid}\t{','.join(chosen)}\n")
    match_pct = n_matched / max(1, len(s1_order)) * 100
    print(f"    → {out_path.name}: {n_matched:,} matched ({match_pct:.1f}%), {len(s1_order)-n_matched:,} singletons")
    return n_matched


def main():
    t0 = time.time()
    print("=" * 80)
    print("V20 POWER INFERENCE  (35 features + multi-threshold sweep)")
    print("=" * 80)

    bundle   = joblib.load("model_finetuned_v20.pkl")
    model    = bundle["model"]
    base_th  = bundle["base_threshold"]
    sib_th   = bundle["sibling_threshold"]
    print(f"Model | BaseTh={base_th:.2f} SibTh={sib_th:.2f} | offline F0.5={bundle['macro_f05']:.4f}")

    BASE_DIR        = Path("student_resource/dataset/test")
    CAND_FILE       = Path("output/candidate_pairs.tsv")
    RAW_SCORES_FILE = Path("output/raw_scores_v20.tsv")
    OUT_DIR         = Path("output/threshold_sweep")
    OUT_DIR.mkdir(exist_ok=True)

    # ── Load candidates ────────────────────────────────────────────────────────
    print("[1/4] Reading candidate pairs...")
    cand_dict   = {}
    needed_tgts = set()
    s1_order    = []
    with open(CAND_FILE,"r",encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            qid = p[0]; s1_order.append(qid)
            cids = [x.strip() for x in p[1].split(",") if x.strip()] if len(p)>1 and p[1] else []
            cand_dict[qid] = cids
            needed_tgts.update(cids)
    print(f"  {len(s1_order):,} queries | {len(needed_tgts):,} unique targets")

    # ── Load targets ───────────────────────────────────────────────────────────
    print("[2/4] Loading target records...")
    tgt_raw = {}
    for fn in ["test_source2.tsv","test_source3.tsv"]:
        path = BASE_DIR / fn
        if not path.exists(): continue
        with open(path,"r",encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                eid = p[0]
                if eid in needed_tgts:
                    tgt_raw[eid] = (p[1], p[2] if len(p)>2 else "", p[3].strip() if len(p)>3 else "")
    print(f"  {len(tgt_raw):,} targets ({time.time()-t0:.1f}s)")

    print("[3/4] Loading Source 1 queries...")
    s1_norm = {}
    with open(BASE_DIR/"test_source1.tsv","r",encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            qid = p[0]
            s1_norm[qid] = (normalize_name(p[1] if len(p)>1 else ""),
                            normalize_address(p[2] if len(p)>2 else "", p[3].strip() if len(p)>3 else ""))
    print(f"  {len(s1_norm):,} queries normalized")

    # ── Scoring loop (saves raw scores) ───────────────────────────────────────
    print("[4/4] Streaming batch inference + saving raw scores...")
    # score_dict: qid → sorted list of (tid, prob) descending
    score_dict  = {}
    CHUNK_SIZE  = 50000
    n_total     = len(s1_order)
    t_score     = time.time()

    for chunk_start in range(0, n_total, CHUNK_SIZE):
        chunk_end  = min(n_total, chunk_start + CHUNK_SIZE)
        chunk_qids = s1_order[chunk_start:chunk_end]

        chunk_tgts = set()
        for qid in chunk_qids:
            for cid in cand_dict[qid]:
                if cid in tgt_raw: chunk_tgts.add(cid)
        chunk_tgt_norm = {}
        for cid in chunk_tgts:
            name,addr,ctry = tgt_raw[cid]
            chunk_tgt_norm[cid] = (normalize_name(name), normalize_address(addr,ctry))

        batch_feats = []; batch_counts = []; batch_cids = []
        for qid in chunk_qids:
            qn,qa = s1_norm[qid]; cids = cand_dict[qid]; cnt = 0; these_cids = []
            for cid in cids:
                if cid in chunk_tgt_norm:
                    tn,ta = chunk_tgt_norm[cid]
                    batch_feats.append(compute_pair_features(qn,qa,tn,ta))
                    these_cids.append(cid); cnt += 1
            batch_counts.append(cnt); batch_cids.append(these_cids)

        if batch_feats:
            X = np.array(batch_feats, dtype=np.float32)
            probs = model.predict_proba(X)[:,1]
            offset = 0
            for i, qid in enumerate(chunk_qids):
                cnt   = batch_counts[i]
                cids  = batch_cids[i]
                if cnt == 0:
                    score_dict[qid] = []
                else:
                    q_probs = probs[offset:offset+cnt]
                    pairs = sorted(zip(cids, q_probs.tolist()), key=lambda x: -x[1])
                    score_dict[qid] = pairs
                    offset += cnt
        else:
            for qid in chunk_qids:
                score_dict[qid] = []

        del chunk_tgt_norm, batch_feats, batch_cids
        done    = chunk_end
        elapsed = time.time() - t_score
        rate    = done / max(1.0, elapsed)
        print(f"  {done:,}/{n_total:,} ({done/n_total*100:.1f}%) | {rate:.0f} q/s | {elapsed:.1f}s")

    print(f"Inference done in {time.time()-t0:.1f}s")

    # Save raw scores
    print(f"Saving raw scores → {RAW_SCORES_FILE}")
    with open(RAW_SCORES_FILE,"w",encoding="utf-8") as f:
        f.write("source1_entity_id\tscored_candidates\n")
        for qid in s1_order:
            pairs = score_dict[qid]
            val   = ";".join(f"{tid}:{prob:.4f}" for tid,prob in pairs)
            f.write(f"{qid}\t{val}\n")
    print(f"  Raw scores saved ({RAW_SCORES_FILE.stat().st_size//1024//1024} MB)")

    # ── Multi-threshold sweep ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("THRESHOLD SWEEP  (BaseTh 0.30 → 0.90, SibTh = BaseTh - 0.05)")
    print("=" * 80)

    thresholds = [round(x, 2) for x in np.arange(0.30, 0.92, 0.05)]
    results = []

    for th in thresholds:
        sib = max(0.25, th - 0.05)
        out = OUT_DIR / f"submission_th{int(th*100):02d}.tsv"
        n_matched = apply_threshold(score_dict, s1_order, th, sib, out)
        match_pct = n_matched / max(1, len(s1_order)) * 100
        results.append((th, sib, n_matched, match_pct, out))

    # The optimal threshold for F0.5:
    # F0.5 rewards precision 4×. A 60-70% match rate is likely ideal.
    # Use model's grid-search recommended threshold as primary
    print("\nUsing model's optimal threshold for final submission...")
    opt_out = OUT_DIR / f"submission_th{int(base_th*100):02d}.tsv"
    if not opt_out.exists():
        apply_threshold(score_dict, s1_order, base_th, sib_th, opt_out)

    # Write final output/matching_results.tsv = model's best threshold
    FINAL = Path("output/matching_results.tsv")
    FINAL_V20 = Path("output/matching_results_v20.tsv")
    apply_threshold(score_dict, s1_order, base_th, sib_th, FINAL)
    apply_threshold(score_dict, s1_order, base_th, sib_th, FINAL_V20)
    print(f"\nFinal submission → {FINAL} (BaseTh={base_th:.2f}, SibTh={sib_th:.2f})")

    print("\n" + "=" * 80)
    print("THRESHOLD SWEEP SUMMARY (choose the one to submit based on score target):")
    print(f"{'BaseTh':>8} {'SibTh':>7} {'Matched':>12} {'MatchRate':>10}")
    print("-" * 45)
    for th, sib, nm, mpct, out in results:
        marker = " ← MODEL OPTIMAL" if abs(th - base_th) < 0.01 else ""
        print(f"{th:>8.2f} {sib:>7.2f} {nm:>12,} {mpct:>9.1f}%{marker}")
    print("=" * 80)
    print(f"All threshold TSVs in: {OUT_DIR}")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
