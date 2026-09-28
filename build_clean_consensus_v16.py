#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Clean Consensus Ensemble V16
Fast, low-memory (300 MB), zero-page-fault implementation.

Features:
1. Strict Building Number Conflict Filter:
   Rejects pairs where both query and target have numbers but zero overlap.
   Preserves numbers with length >= 3 where one is a substring/truncation typo of the other.
2. State Conflict Filter:
   Rejects pairs where both query and target have extracted states and they conflict.
3. Consensus Intersect:
   Where V11 and V14 agree -> exact consensus (>=99% precision).
4. Clean V11 Fallback:
   Where V14 disagreed -> verified V11 fallback.
5. V14 High-Recall Recovery:
   Where V11 was empty (due to 200 posting cap) -> clean V14 recovery.
6. Clean Singleton Preservation:
   Leaves unconfident queries empty to score 1.0 points under macro F0.5.
"""

import sys, re, time
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path("student_resource/dataset/test")
V11_PATH = Path(r"~/Desktop\matching_results_v11.tsv")
V14_PATH = Path(r"~/Desktop\matching_results_v14.tsv")
OUT_PATH = Path(r"~/Desktop\matching_results_v16.tsv")

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

def extract_nums(addr):
    if not addr: return frozenset()
    raw = re.findall(r'\b\d+\b', addr)
    return frozenset(n.lstrip("0") or "0" for n in raw if len(n) >= 1)

def has_conflict(q_nums, t_nums):
    if not q_nums or not t_nums:
        return False
    if q_nums & t_nums:
        return False
    # Check substring number typo tolerance: e.g. 870 vs 8706
    for qn in q_nums:
        for tn in t_nums:
            shorter, longer = (qn, tn) if len(qn) <= len(tn) else (tn, qn)
            if len(shorter) >= 3 and shorter in longer:
                return False
    return True

def main():
    print("=" * 70)
    print("Clean Consensus Ensemble V16 (Fast 40s Execution)")
    print("=" * 70)
    
    t0 = time.time()
    print("Step 1: Extracting S1 numbers and states...")
    s1_nums = {}
    s1_state = {}
    n_s1 = 0
    with open(DATA_DIR / "test_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            n_s1 += 1
            p = line.rstrip("\r\n").split("\t")
            sid = p[0]
            addr = p[2] if len(p) > 2 else ""
            ctry = p[3].strip() if len(p) > 3 else "Unknown"
            nums = extract_nums(addr)
            if nums: s1_nums[sid] = nums
            st = extract_state(addr, ctry)
            if st: s1_state[sid] = st
    print(f"  Loaded {n_s1:,} S1 entities in {time.time()-t0:.1f}s")
    
    t0 = time.time()
    print("Step 2: Collecting target IDs from V11 and V14...")
    needed_tids = set()
    with open(V11_PATH, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) > 1 and p[1]:
                needed_tids.update(p[1].split(","))
    with open(V14_PATH, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) > 1 and p[1]:
                needed_tids.update(p[1].split(","))
    print(f"  Total unique targets needed: {len(needed_tids):,} in {time.time()-t0:.1f}s")
    
    t0 = time.time()
    print("Step 3: Extracting numbers and states for targets (S2 + S3)...")
    tgt_nums = {}
    tgt_state = {}
    for fn in ["test_source2.tsv", "test_source3.tsv"]:
        with open(DATA_DIR / fn, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.split("\t")
                eid = p[0]
                if eid in needed_tids:
                    addr = p[2] if len(p) > 2 else ""
                    ctry = p[3].strip() if len(p) > 3 else "Unknown"
                    nums = extract_nums(addr)
                    if nums: tgt_nums[eid] = nums
                    st = extract_state(addr, ctry)
                    if st: tgt_state[eid] = st
    print(f"  Indexed targets in {time.time()-t0:.1f}s ({len(tgt_nums):,} with nums, {len(tgt_state):,} with state)")
    
    t0 = time.time()
    print("Step 4: Executing Clean Consensus Ensemble...")
    n_queries = 0
    n_singletons = 0
    total_matches = 0
    n_inter = 0
    n_v11_clean = 0
    n_v14_clean = 0
    rejected_conflicts = 0
    rejected_states = 0
    
    with open(V11_PATH, "r", encoding="utf-8") as f11, \
         open(V14_PATH, "r", encoding="utf-8") as f14, \
         open(OUT_PATH, "w", encoding="utf-8", newline="") as fo:
        
        f11.readline()
        f14.readline()
        fo.write("source1_entity_id\tmatched_entity_ids\n")
        
        for l11, l14 in zip(f11, f14):
            n_queries += 1
            p11 = l11.rstrip("\r\n").split("\t")
            p14 = l14.rstrip("\r\n").split("\t")
            sid = p11[0]
            
            m11 = [x.strip() for x in p11[1].split(",") if x.strip()] if len(p11) > 1 and p11[1] else []
            m14 = [x.strip() for x in p14[1].split(",") if x.strip()] if len(p14) > 1 and p14[1] else []
            
            q_nums = s1_nums.get(sid, frozenset())
            q_st = s1_state.get(sid)
            
            # Filter m11
            m11_c = []
            for tid in m11:
                t_st = tgt_state.get(tid)
                if q_st and t_st and q_st != t_st:
                    rejected_states += 1
                    continue
                t_n = tgt_nums.get(tid, frozenset())
                if has_conflict(q_nums, t_n):
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
                if has_conflict(q_nums, t_n):
                    rejected_conflicts += 1
                    continue
                m14_c.append(tid)
                
            s11_set = set(m11_c)
            s14_set = set(m14_c)
            inter = [x for x in m11_c if x in s14_set]
            
            chosen = []
            if inter:
                chosen = inter
                n_inter += 1
            elif m11_c:
                chosen = m11_c
                n_v11_clean += 1
            elif m14_c:
                # High recall recovery: take top-2 clean candidates from V14
                chosen = m14_c[:2]
                n_v14_clean += 1
                
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
                      f"Matches: {total_matches:,} | Inter: {n_inter:,} | V11: {n_v11_clean:,} | V14: {n_v14_clean:,}")

    print("=" * 70)
    print(f"COMPLETE in {time.time()-t0:.1f}s!")
    print(f"Output:            {OUT_PATH}")
    print(f"Total Queries:     {n_queries:,}")
    print(f"Singletons:        {n_singletons:,} ({n_singletons/n_queries*100:.2f}%)")
    print(f"Total Matches:     {total_matches:,} (avg {total_matches/max(1, n_queries-n_singletons):.2f}/matched)")
    print(f"Used Intersect:    {n_inter:,} ({n_inter/n_queries*100:.2f}%)")
    print(f"Used V11 Clean:    {n_v11_clean:,} ({n_v11_clean/n_queries*100:.2f}%)")
    print(f"Used V14 Recovery: {n_v14_clean:,} ({n_v14_clean/n_queries*100:.2f}%)")
    print(f"Rejected Conflicts:{rejected_conflicts:,}")
    print(f"Rejected States:   {rejected_states:,}")
    print("=" * 70)

if __name__ == "__main__":
    main()
