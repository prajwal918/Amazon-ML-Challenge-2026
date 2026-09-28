#!/usr/bin/env python3
"""
Inspect False Negatives: why were true matches in cands rejected by the matching model?
"""

import sys, re
from collections import defaultdict
from pathlib import Path
from rapidfuzz import fuzz
from anyascii import anyascii

from eval_grandmaster_090 import (
    TRAIN_DIR, prep_record, smart_has_conflict
)

# Load 500 GT queries
gt = {}
with open(TRAIN_DIR / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 500: break
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
for fn in ["train_source2.tsv", "train_source3.tsv"]:
    with open(TRAIN_DIR / fn, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            eid = p[0]
            if eid in needed_tgts:
                target_entities[eid] = (p[1], p[2] if len(p) > 2 else "", p[3].strip() if len(p) > 3 else "")
                
s1_prep = {sid: prep_record(n, a, c) for sid, (n, a, c) in s1_entities.items()}
tgt_prep = {tid: prep_record(n, a, c) for tid, (n, a, c) in target_entities.items()}

# Analyze the feature distributions of TRUE MATCHES
missed_reasons = defaultdict(int)
sample_misses = []

for sid, true_set in gt.items():
    if not true_set: continue
    q = s1_prep[sid]
    for tid in true_set:
        if tid not in tgt_prep: continue
        t = tgt_prep[tid]
        
        # Check why it might be rejected
        has_st_conflict = bool(q["state"] and t["state"] and q["state"] != t["state"])
        has_num_conflict = smart_has_conflict(q["nums"], q["postal"], t["nums"], t["postal"])
        num_agree = bool(q["nums"] and t["nums"] and (q["nums"] & t["nums"]))
        
        core_sort = fuzz.token_sort_ratio(q["core"], t["core"]) if (q["core"] and t["core"]) else 0
        asim = fuzz.token_sort_ratio(q["addr"], t["addr"]) if (q["addr"] and t["addr"]) else 0
        street_sim = fuzz.token_set_ratio(q["addr_nonum"], t["addr_nonum"]) if (q["addr_nonum"] and t["addr_nonum"]) else 0
        is_exact_comp = bool(q["comp"] and t["comp"] and q["comp"] == t["comp"])
        
        q_toks = q["tokens"]
        t_toks = t["tokens"]
        tok_overlap = len(q_toks & t_toks)
        n_j = tok_overlap / max(1, len(q_toks | t_toks))
        
        # Did it match under current rules?
        passed = False
        if not has_st_conflict and not has_num_conflict:
            if is_exact_comp: passed = True
            elif not t["is_latin"] and q["ctry"] == "India" and num_agree and (street_sim >= 60 or asim >= 55): passed = True
            elif num_agree and street_sim >= 65 and (core_sort >= 40 or tok_overlap >= 1): passed = True
            elif core_sort >= 85 and (num_agree or street_sim >= 30 or not t["addr"] or not q["addr"]): passed = True
            elif core_sort >= 72 and (num_agree or street_sim >= 75): passed = True
            
        if not passed:
            if has_st_conflict: missed_reasons["state_conflict"] += 1
            elif has_num_conflict: missed_reasons["number_conflict"] += 1
            elif core_sort < 40 and not is_exact_comp: missed_reasons["low_core_name_sim"] += 1
            elif not num_agree and street_sim < 60: missed_reasons["low_street_sim"] += 1
            else: missed_reasons["other_threshold"] += 1
            
            if len(sample_misses) < 5:
                sample_misses.append({
                    "sid": sid, "tid": tid, "q_name": q["name"], "t_name": t["name"],
                    "q_addr": q["addr"], "t_addr": t["addr"], "core_sort": core_sort,
                    "street_sim": street_sim, "num_agree": num_agree,
                    "has_num_conf": has_num_conflict, "has_st_conf": has_st_conflict
                })

print("=" * 70)
print("MISS REASON DISTRIBUTION ON 500 GT QUERIES:")
print("=" * 70)
for reason, count in sorted(missed_reasons.items(), key=lambda x: -x[1]):
    print(f"  {reason:<25}: {count:,}")
    
print("\nSAMPLE MISSED TRUE MATCHES:")
for m in sample_misses:
    print("-" * 50)
    print(f"Q: [{m['sid']}] Name: '{m['q_name']}' | Addr: '{m['q_addr']}'")
    print(f"T: [{m['tid']}] Name: '{m['t_name']}' | Addr: '{m['t_addr']}'")
    print(f"   core_sort={m['core_sort']}, street_sim={m['street_sim']}, num_agree={m['num_agree']}, num_conf={m['has_num_conf']}, st_conf={m['has_st_conf']}")
print("=" * 70)
