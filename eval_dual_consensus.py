#!/usr/bin/env python3
"""
Dual Consensus Pipeline on 5,000 Ground Truth Queries.
Tests:
1. Strict Intersect
2. Consensus Intersect else Fallback
3. Filtered Smart Union
4. Singleton-Protected Calibrated Union
Target: Macro F0.5 >= 0.90+.
"""

import sys, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from rapidfuzz import fuzz
from anyascii import anyascii

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from eval_grandmaster_090 import (
    TRAIN_DIR, compute_entity_f_beta, prep_record, smart_has_conflict
)

def main():
    print("=" * 70)
    print("Multi-Strategy Ensemble Evaluation (5,000 Queries + 50k Targets)")
    print("=" * 70)
    
    t0 = time.time()
    # 1. Load 5,000 GT queries
    gt = {}
    with open(TRAIN_DIR / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i >= 5000: break
            p = line.rstrip("\r\n").split("\t")
            gt[p[0]] = set(x.strip() for x in p[1].split(",") if x.strip()) if len(p) > 1 and p[1] else set()
            
    needed_s1 = set(gt.keys())
    needed_tgts = {tid for tgts in gt.values() for tid in tgts}
    n_singletons = sum(1 for tgts in gt.values() if len(tgts) == 0)
    
    # 2. Load S1 entities
    s1_entities = {}
    with open(TRAIN_DIR / "train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in needed_s1:
                s1_entities[p[0]] = (p[1], p[2] if len(p) > 2 else "", p[3].strip() if len(p) > 3 else "")
                
    # 3. Load Targets + 50,000 distractors
    target_entities = {}
    distractors_loaded = 0
    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        with open(TRAIN_DIR / fn, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                eid = p[0]
                if eid in needed_tgts:
                    target_entities[eid] = (p[1], p[2] if len(p) > 2 else "", p[3].strip() if len(p) > 3 else "")
                elif distractors_loaded < 50000:
                    target_entities[eid] = (p[1], p[2] if len(p) > 2 else "", p[3].strip() if len(p) > 3 else "")
                    distractors_loaded += 1
                    
    print(f"Loaded {len(gt):,} queries, {len(target_entities):,} targets in {time.time()-t0:.1f}s")
    
    s1_prep = {sid: prep_record(n, a, c) for sid, (n, a, c) in s1_entities.items()}
    tgt_prep = {tid: prep_record(n, a, c) for tid, (n, a, c) in target_entities.items()}
    
    comp_idx = defaultdict(list)
    token_idx = defaultdict(list)
    num_addr_idx = defaultdict(list)
    
    for tid, t in tgt_prep.items():
        if t["comp"] and len(t["comp"]) >= 4:
            comp_idx[(t["ctry"], t["comp"])].append(tid)
        for tok in t["tokens"]:
            if len(tok) >= 3:
                token_idx[(t["ctry"], tok)].append(tid)
        for num in t["nums"]:
            if len(num) >= 2:
                num_addr_idx[(t["ctry"], num)].append(tid)
                
    preds_a = {}
    preds_b = {}
    
    for sid, q in s1_prep.items():
        q_ctry = q["ctry"]
        cands = set()
        if q["comp"] and (q_ctry, q["comp"]) in comp_idx:
            cands.update(comp_idx[(q_ctry, q["comp"])])
        for tok in q["tokens"]:
            key = (q_ctry, tok)
            if key in token_idx and len(token_idx[key]) <= 400:
                cands.update(token_idx[key])
        for num in q["nums"]:
            key = (q_ctry, num)
            if key in num_addr_idx and len(num_addr_idx[key]) <= 250:
                cands.update(num_addr_idx[key])
                
        m_a = []
        m_b = []
        
        for tid in cands:
            t = tgt_prep[tid]
            if t["ctry"] != q_ctry: continue
            if q["state"] and t["state"] and q["state"] != t["state"]: continue
            if smart_has_conflict(q["nums"], q["postal"], t["nums"], t["postal"]): continue
            
            num_agree = bool(q["nums"] and t["nums"] and (q["nums"] & t["nums"]))
            core_sort = fuzz.token_sort_ratio(q["core"], t["core"]) if (q["core"] and t["core"]) else 0
            asim = fuzz.token_sort_ratio(q["addr"], t["addr"]) if (q["addr"] and t["addr"]) else 0
            street_sim = fuzz.token_set_ratio(q["addr_nonum"], t["addr_nonum"]) if (q["addr_nonum"] and t["addr_nonum"]) else 0
            is_exact_comp = bool(q["comp"] and t["comp"] and q["comp"] == t["comp"])
            
            q_toks = q["tokens"]
            t_toks = t["tokens"]
            n_j = len(q_toks & t_toks) / max(1, len(q_toks | t_toks))
            
            q_atoks = set(re.findall(r"\b[a-z]{2,}\b", q["addr"]))
            t_atoks = set(re.findall(r"\b[a-z]{2,}\b", t["addr"]))
            a_j = len(q_atoks & t_atoks) / max(1, len(q_atoks | t_atoks))
            
            # Model A: Token Jaccard + Building Overlap (Conservative)
            is_a = False
            if a_j >= 0.50 or (num_agree and a_j >= 0.28): is_a = True
            elif n_j >= 0.70: is_a = True
            elif a_j >= 0.35 and n_j >= 0.35: is_a = True
            if is_a: m_a.append((max(n_j, a_j), tid))
            
            # Model B: String Metric Core Sort + Levenshtein (Permutation-Resilient)
            is_b = False
            if is_exact_comp:
                if not t["addr"] or not q["addr"] or street_sim >= 20 or num_agree or core_sort >= 50:
                    is_b = True
            elif not t["is_latin"] and q_ctry == "India":
                if num_agree and (street_sim >= 60 or asim >= 55): is_b = True
            elif num_agree and street_sim >= 65 and (core_sort >= 40 or len(q_toks & t_toks) >= 1):
                is_b = True
            elif core_sort >= 85 and (num_agree or street_sim >= 30 or not t["addr"] or not q["addr"]):
                is_b = True
            elif core_sort >= 72 and (num_agree or street_sim >= 75):
                is_b = True
            if is_b: m_b.append((core_sort, tid))
            
        m_a.sort(reverse=True)
        m_b.sort(reverse=True)
        preds_a[sid] = [tid for _, tid in m_a[:5]]
        preds_b[sid] = [tid for _, tid in m_b[:5]]
        
    def evaluate_preds(name, preds_dict):
        scores = []
        tp = fp = fn = 0
        correct_singletons = 0
        for sid, true_m in gt.items():
            p = set(preds_dict[sid])
            cur_tp = len(p & true_m)
            cur_fp = len(p - true_m)
            cur_fn = len(true_m - p)
            tp += cur_tp; fp += cur_fp; fn += cur_fn
            if len(true_m) == 0 and len(p) == 0:
                correct_singletons += 1
            scores.append(compute_entity_f_beta(p, true_m, beta=0.5))
        macro_f = np.mean(scores)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        print(f"{name:<35}: F0.5 = {macro_f:.4f} | Prec = {prec:.4f} | Rec = {rec:.4f} | TP = {tp:,} | FP = {fp:,} | Sg = {correct_singletons}/{n_singletons}")
        return macro_f
        
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON RESULTS:")
    print("=" * 70)
    
    # 1. Model A Standalone
    evaluate_preds("Model A (Token Jaccard V11)", preds_a)
    
    # 2. Model B Standalone
    evaluate_preds("Model B (String Metric V14)", preds_b)
    
    # 3. Strict Intersection
    preds_inter = {sid: list(set(preds_a[sid]) & set(preds_b[sid])) for sid in gt}
    evaluate_preds("Strategy 1: Strict Intersection", preds_inter)
    
    # 4. Consensus Intersect else A
    preds_inter_a = {sid: list(set(preds_a[sid]) & set(preds_b[sid])) if (set(preds_a[sid]) & set(preds_b[sid])) else preds_a[sid] for sid in gt}
    evaluate_preds("Strategy 2: Intersect else Model A", preds_inter_a)
    
    # 5. Full Union
    preds_union = {sid: list(dict.fromkeys(preds_a[sid] + preds_b[sid]))[:4] for sid in gt}
    evaluate_preds("Strategy 3: Full Union (Capped 4)", preds_union)
    
    # 6. High-Precision Calibrated Union
    preds_calibrated = {}
    for sid in gt:
        sa = set(preds_a[sid])
        sb = set(preds_b[sid])
        chosen = list(preds_a[sid])
        
        # Add from B only if confident
        q_comp = s1_prep[sid]["comp"]
        q_core = s1_prep[sid]["core"]
        for tid in preds_b[sid]:
            if tid not in chosen and len(chosen) < 4:
                t_comp = tgt_prep[tid]["comp"]
                t_core = tgt_prep[tid]["core"]
                is_exact = bool(q_comp and t_comp and q_comp == t_comp)
                c_sort = fuzz.token_sort_ratio(q_core, t_core) if (q_core and t_core) else 0
                if is_exact or c_sort >= 40:
                    chosen.append(tid)
        preds_calibrated[sid] = chosen[:4]
        
    score_calibrated = evaluate_preds("Strategy 4: Calibrated Union V17", preds_calibrated)
    print("=" * 70)
    
    if score_calibrated >= 0.90:
        print(f"\n--> SUCCESS: Confirmed Macro F0.5 = {score_calibrated:.4f} >= 0.90!")

if __name__ == "__main__":
    main()
