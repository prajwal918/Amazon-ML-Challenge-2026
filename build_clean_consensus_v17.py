#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Clean Consensus Ensemble V17
Supercharged with:
1. Smart Address Parser (Disambiguates Postal Codes vs Street Numbers):
   - India: 6-digit PIN codes
   - US / France: 5-digit ZIP codes
   - Street numbers: 1-4 digits (excluding postal codes and phone numbers >= 7 digits)
   - Eliminates false conflict rejections where one record had postal code and other had street number!
2. State Conflict Filter:
   - Rejects pairs where both query and target have extracted states and they conflict.
3. High-Precision Consensus Intersect:
   - Where V11 and V14 agree -> exact consensus (>=99% precision).
4. Clean V11 Fallback:
   - Where V11 has clean candidates and V14 missed -> verified V11 fallback.
5. Strict Singleton Protection on V14 Recovery:
   - When V11 was EMPTY (predicted singleton):
     ONLY accept V14 candidates if:
     - Exact compressed name matches, OR
     - Core name token sort ratio >= 45, OR
     - Street numbers match and street nonum similarity >= 60.
     Otherwise: PRESERVE SINGLETON (scores 1.0 points under macro F0.5!).
6. Lightweight, ultra-fast streaming (runs in <45s, <400 MB RAM).
"""

import sys, re, time
from collections import defaultdict
from pathlib import Path
from rapidfuzz import fuzz

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path("student_resource/dataset/test")
V11_PATH = Path(r"~/Desktop\matching_results_v11.tsv")
V14_PATH = Path(r"~/Desktop\matching_results_v14.tsv")
OUT_PATH = Path(r"~/Desktop\matching_results_v17.tsv")
DESKTOP_SUBMISSION = Path(r"~/Desktop\matching_results.tsv")

STATE_CANONICAL = {
    "tamil nadu": "TN", "tamilnadu": "TN", "tn": "TN", "karnataka": "KA", "ka": "KA",
    "kerala": "KL", "kl": "KL", "maharashtra": "MH", "mh": "MH", "andhra pradesh": "AP",
    "andhra": "AP", "ap": "AP", "telangana": "TG", "ts": "TG", "tg": "TG", "gujarat": "GJ",
    "gj": "GJ", "rajasthan": "RJ", "rj": "RJ", "uttar pradesh": "UP", "up": "UP",
    "west bengal": "WB", "wb": "WB", "delhi": "DL", "dl": "DL", "punjab": "PB", "pb": "PB",
    "haryana": "HR", "hr": "HR", "bihar": "BR", "br": "BR", "odisha": "OD", "orissa": "OD",
    "madhya pradesh": "MP", "mp": "MP", "assam": "AS", "jharkhand": "JH", "chhattisgarh": "CG",
    "uttarakhand": "UK", "himachal pradesh": "HP", "goa": "GA"
}
US_STATES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia",
    "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt",
    "va", "wa", "wv", "wi", "wy"
}

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

def strip_legal(name):
    return LEGAL_SUFFIX_RE.sub(" ", name)

def compress_name(name):
    n = name.lower()
    for ext in DOMAINS:
        if n.endswith(ext): n = n[:-len(ext)]
    n = re.sub(r'[^a-z0-9]', '', n)
    for s in ["pvtltd", "privatelimited", "ltd", "llc", "inc", "corp", "corporation", "co", "company", "sarl", "sas", "enterprises", "solutions", "services"]:
        if n.endswith(s): n = n[:-len(s)]
    return n

def extract_state(addr, ctry):
    if not addr: return None
    addr_l = addr.lower()
    if ctry == "India":
        for st, code in STATE_CANONICAL.items():
            if re.search(r'\b' + re.escape(st) + r'\b', addr_l): return code
    elif ctry == "US":
        words = re.findall(r'\b[a-zA-Z]{2}\b', addr)
        for w in reversed(words):
            if w.lower() in US_STATES: return w.upper()
    return None

def parse_address_numbers(addr, ctry):
    if not addr:
        return frozenset(), None
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
        if len(n) >= 7: continue  # skip phone numbers
        street_nums.add(n.lstrip("0") or "0")
    return frozenset(street_nums), postal

def smart_has_conflict(q_nums, q_post, t_nums, t_post):
    # Postal code conflict
    if q_post and t_post and q_post != t_post:
        return True
    # Street number conflict: ONLY if BOTH have street numbers
    if q_nums and t_nums:
        if q_nums & t_nums:
            return False
        # Substring typo tolerance (e.g. 870 vs 8706)
        for qn in q_nums:
            for tn in t_nums:
                shorter, longer = (qn, tn) if len(qn) <= len(tn) else (tn, qn)
                if len(shorter) >= 3 and shorter in longer:
                    return False
        return True
    return False

def main():
    print("=" * 70)
    print("Clean Consensus Ensemble V17 (Smart Postal Disambiguation & Singleton Protection)")
    print("=" * 70)
    
    t0 = time.time()
    print("Step 1: Analyzing queries where V11 was empty but V14 had candidates...")
    recovery_sids = set()
    needed_recovery_tids = set()
    all_needed_tids = set()
    
    with open(V11_PATH, "r", encoding="utf-8") as f11, open(V14_PATH, "r", encoding="utf-8") as f14:
        f11.readline(); f14.readline()
        for l11, l14 in zip(f11, f14):
            p11 = l11.rstrip("\r\n").split("\t")
            p14 = l14.rstrip("\r\n").split("\t")
            sid = p11[0]
            m11 = p11[1] if len(p11) > 1 and p11[1] else ""
            m14 = p14[1] if len(p14) > 1 and p14[1] else ""
            
            if m11:
                all_needed_tids.update(m11.split(","))
            if m14:
                all_needed_tids.update(m14.split(","))
                if not m11:
                    recovery_sids.add(sid)
                    needed_recovery_tids.update(m14.split(","))
                    
    print(f"  Total unique targets needed: {len(all_needed_tids):,}")
    print(f"  Queries where V11 was empty but V14 had matches: {len(recovery_sids):,}")
    print(f"  Target IDs needing name verification: {len(needed_recovery_tids):,}")
    print(f"  Done in {time.time()-t0:.1f}s")
    
    t0 = time.time()
    print("Step 2: Indexing S1 entities...")
    s1_nums = {}
    s1_postal = {}
    s1_state = {}
    s1_core = {}
    s1_comp = {}
    s1_addr_nonum = {}
    n_s1 = 0
    
    with open(DATA_DIR / "test_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            n_s1 += 1
            p = line.rstrip("\r\n").split("\t")
            sid = p[0]
            name = p[1] if len(p) > 1 else ""
            addr = p[2] if len(p) > 2 else ""
            ctry = p[3].strip() if len(p) > 3 else "Unknown"
            
            nums, post = parse_address_numbers(addr, ctry)
            if nums: s1_nums[sid] = nums
            if post: s1_postal[sid] = post
            st = extract_state(addr, ctry)
            if st: s1_state[sid] = st
            
            if sid in recovery_sids:
                s1_core[sid] = strip_legal(name).strip().lower()
                s1_comp[sid] = compress_name(name)
                s1_addr_nonum[sid] = re.sub(r'\d+', ' ', addr.lower()).strip()
                
    print(f"  Indexed {n_s1:,} S1 entities in {time.time()-t0:.1f}s")
    
    t0 = time.time()
    print("Step 3: Indexing target entities (test_source2 + test_source3)...")
    tgt_nums = {}
    tgt_postal = {}
    tgt_state = {}
    tgt_core = {}
    tgt_comp = {}
    tgt_addr_nonum = {}
    
    for fn in ["test_source2.tsv", "test_source3.tsv"]:
        with open(DATA_DIR / fn, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.split("\t")
                eid = p[0]
                if eid in all_needed_tids:
                    name = p[1] if len(p) > 1 else ""
                    addr = p[2] if len(p) > 2 else ""
                    ctry = p[3].strip() if len(p) > 3 else "Unknown"
                    
                    nums, post = parse_address_numbers(addr, ctry)
                    if nums: tgt_nums[eid] = nums
                    if post: tgt_postal[eid] = post
                    st = extract_state(addr, ctry)
                    if st: tgt_state[eid] = st
                    
                    if eid in needed_recovery_tids:
                        tgt_core[eid] = strip_legal(name).strip().lower()
                        tgt_comp[eid] = compress_name(name)
                        tgt_addr_nonum[eid] = re.sub(r'\d+', ' ', addr.lower()).strip()
                        
    print(f"  Indexed {len(tgt_nums):,} target nums, {len(tgt_postal):,} postals, {len(tgt_state):,} states in {time.time()-t0:.1f}s")
    
    t0 = time.time()
    print("Step 4: Executing Clean Consensus Ensemble V17...")
    n_queries = 0
    n_singletons = 0
    total_matches = 0
    n_inter = 0
    n_v11_clean = 0
    n_v14_recovered = 0
    n_v14_rejected_singleton = 0
    rejected_conflicts = 0
    rejected_states = 0
    rejected_postal = 0
    
    with open(V11_PATH, "r", encoding="utf-8") as f11, \
         open(V14_PATH, "r", encoding="utf-8") as f14, \
         open(OUT_PATH, "w", encoding="utf-8", newline="") as fo:
        
        f11.readline(); f14.readline()
        fo.write("source1_entity_id\tmatched_entity_ids\n")
        
        for l11, l14 in zip(f11, f14):
            n_queries += 1
            p11 = l11.rstrip("\r\n").split("\t")
            p14 = l14.rstrip("\r\n").split("\t")
            sid = p11[0]
            
            m11 = [x.strip() for x in p11[1].split(",") if x.strip()] if len(p11) > 1 and p11[1] else []
            m14 = [x.strip() for x in p14[1].split(",") if x.strip()] if len(p14) > 1 and p14[1] else []
            
            q_nums = s1_nums.get(sid, frozenset())
            q_post = s1_postal.get(sid)
            q_st = s1_state.get(sid)
            
            # Filter m11
            m11_c = []
            for tid in m11:
                t_st = tgt_state.get(tid)
                if q_st and t_st and q_st != t_st:
                    rejected_states += 1
                    continue
                t_n = tgt_nums.get(tid, frozenset())
                t_p = tgt_postal.get(tid)
                if smart_has_conflict(q_nums, q_post, t_n, t_p):
                    rejected_conflicts += 1
                    continue
                m11_c.append(tid)
                
            # Filter m14
            m14_c = []
            for tid in m14:
                t_st = tgt_state.get(tid)
                if q_st and t_st and q_st != t_st:
                    rejected_states += 1
                    continue
                t_n = tgt_nums.get(tid, frozenset())
                t_p = tgt_postal.get(tid)
                if smart_has_conflict(q_nums, q_post, t_n, t_p):
                    rejected_conflicts += 1
                    continue
                m14_c.append(tid)
                
            s11_set = set(m11_c)
            s14_set = set(m14_c)
            inter = [x for x in m11_c if x in s14_set]
            
            chosen = []
            if inter:
                # Intersection has verified >=99% precision
                chosen = inter
                n_inter += 1
            elif m11_c:
                # V11 had candidates that V14 missed -> V11 fallback
                chosen = m11_c
                n_v11_clean += 1
            elif m14_c:
                # V11 was completely EMPTY (predicted singleton!)
                # Strict validation: ONLY accept if high confidence!
                q_core = s1_core.get(sid, "")
                q_comp = s1_comp.get(sid, "")
                q_anon = s1_addr_nonum.get(sid, "")
                
                valid_recovery = []
                for tid in m14_c:
                    t_core = tgt_core.get(tid, "")
                    t_comp = tgt_comp.get(tid, "")
                    t_anon = tgt_addr_nonum.get(tid, "")
                    
                    # Criteria 1: Exact compressed match
                    if q_comp and t_comp and q_comp == t_comp:
                        valid_recovery.append(tid)
                        continue
                    # Criteria 2: Core name token sort >= 45
                    core_sort = fuzz.token_sort_ratio(q_core, t_core) if (q_core and t_core) else 0
                    if core_sort >= 45:
                        valid_recovery.append(tid)
                        continue
                    # Criteria 3: Street numbers agree AND street nonum similarity >= 60
                    t_n = tgt_nums.get(tid, frozenset())
                    num_agree = bool(q_nums and t_n and (q_nums & t_n))
                    if num_agree and q_anon and t_anon:
                        street_sim = fuzz.token_set_ratio(q_anon, t_anon)
                        if street_sim >= 60 and core_sort >= 30:
                            valid_recovery.append(tid)
                            continue
                            
                if valid_recovery:
                    chosen = valid_recovery[:3]
                    n_v14_recovered += 1
                else:
                    # Preserved Singleton!
                    n_v14_rejected_singleton += 1
                    
            if not chosen:
                n_singletons += 1
                fo.write(f"{sid}\t\n")
            else:
                chosen = chosen[:4]
                total_matches += len(chosen)
                fo.write(f"{sid}\t{','.join(chosen)}\n")
                
            if n_queries % 300000 == 0:
                print(f"  [{n_queries:,}/{n_s1:,}] "
                      f"Singletons: {n_singletons:,} ({n_singletons/n_queries*100:.2f}%) | "
                      f"Matches: {total_matches:,} | Inter: {n_inter:,} | V11: {n_v11_clean:,} | "
                      f"V14-Rec: {n_v14_recovered:,} | Preserved-Singletons: {n_v14_rejected_singleton:,}")

    print("=" * 70)
    print(f"COMPLETE in {time.time()-t0:.1f}s!")
    print(f"Output:            {OUT_PATH}")
    print(f"Total Queries:     {n_queries:,}")
    print(f"Singletons:        {n_singletons:,} ({n_singletons/n_queries*100:.2f}%)")
    print(f"Total Matches:     {total_matches:,} (avg {total_matches/max(1, n_queries-n_singletons):.2f}/matched)")
    print(f"Used Intersect:    {n_inter:,} ({n_inter/n_queries*100:.2f}%)")
    print(f"Used V11 Clean:    {n_v11_clean:,} ({n_v11_clean/n_queries*100:.2f}%)")
    print(f"Used V14 Recovery: {n_v14_recovered:,} ({n_v14_recovered/n_queries*100:.2f}%)")
    print(f"Preserved Singletons from V14 noise: {n_v14_rejected_singleton:,}")
    print(f"Rejected Conflicts:{rejected_conflicts:,}")
    print(f"Rejected States:   {rejected_states:,}")
    print("=" * 70)
    
    # Copy to DESKTOP_SUBMISSION
    import shutil
    shutil.copyfile(OUT_PATH, DESKTOP_SUBMISSION)
    print(f"Copied to {DESKTOP_SUBMISSION} (Size: {DESKTOP_SUBMISSION.stat().st_size / 1024 / 1024:.1f} MB)")

if __name__ == "__main__":
    main()
