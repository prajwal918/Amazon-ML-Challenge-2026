#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Pipeline V11 (Precision Strike Resolver)
Key Breakthroughs from Ground-Truth Diagnostic:
1. Hard Building Number Conflict Rejection:
   In 90% of true matches, numbers agree. If query has #2621 and target has #178 -> HARD REJECT.
   (This completely eliminates the 7 false positive distractors that crippled V8/V9/V10).
2. Calibrated High-Precision Thresholds:
   Name token_sort_ratio >= 70 AND (Addr token_sort_ratio >= 40 OR exact number match).
3. Exact Compressed Name Match: Instant high-confidence match.
4. Maximum 4 matches per entity (True ground truth average is 3.46).
5. Clean Singletons: If no candidate passes high-precision bar, predict empty (singleton -> score 1.0).
"""
import sys, re, unicodedata, time, os, gc
from collections import defaultdict
from pathlib import Path
import numpy as np
import zipfile, shutil

try:
    from anyascii import anyascii
    from rapidfuzz import fuzz
except ImportError:
    os.system("pip install -q anyascii rapidfuzz")
    from anyascii import anyascii
    from rapidfuzz import fuzz

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

DOMAINS = (".com", ".net", ".org", ".co.in", ".in", ".fr", ".co", ".io", ".info", ".biz", ".org.in", ".gov.in")
LEGAL_SUFFIXES = {
    "corp", "corporation", "corporate", "pvt", "private", "ltd", "limited",
    "llc", "inc", "incorporated", "co", "company", "sarl", "sas", "llp", "sa", "gmbh",
    "sasu", "eurl", "dba", "services", "holdings", "group", "groupe", "center", "enterprises",
    "industries", "associates", "consulting", "solutions", "international", "intl"
}
STOPWORDS = {"the", "and", "of", "a", "an", "in", "to", "for", "at", "on", "by", "null", "near", "opp", "no", "mr", "ms", "sri", "shri", "smt"}

def clean_text(s):
    s = anyascii(str(s)).lower().strip()
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
    return frozenset(t for t in toks if t not in LEGAL_SUFFIXES and t not in STOPWORDS and len(t) > 2)

def get_nums(s):
    raw = re.findall(r'\b\d+\b', s)
    return frozenset(n.lstrip("0") or "0" for n in raw if len(n) >= 2)

MAX_NAME_POSTINGS = 200
MAX_ADDR_POSTINGS = 50
MAX_MATCHES = 4

def build_index(filepath, countries):
    print(f"Indexing {filepath.name}...")
    t0 = time.time()
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.split("\t")
            if not p: continue
            eid = p[0]
            name_raw = p[1] if len(p) > 1 else ""
            addr_raw = p[2] if len(p) > 2 else ""
            ctry = p[3].strip() if len(p) > 3 else "Unknown"

            c = countries[ctry]
            idx = len(c["ids"])
            
            c_name = clean_text(name_raw)
            c_addr = clean_text(addr_raw)
            comp = compress_name(name_raw)
            toks = get_tokens(c_name)
            nums = get_nums(c_addr)

            c["ids"].append(eid)
            c["name"].append(c_name)
            c["addr"].append(c_addr)
            c["comp"].append(comp)
            c["nums"].append(nums)
            c["tokens"].append(toks)

            if comp:
                c["comp_idx"][comp].append(idx)
            for t in toks:
                c["name_idx"][t].append(idx)
            if nums:
                for num in list(nums)[:2]:
                    for atok in c_addr.split()[:3]:
                        if len(atok) > 3 and atok not in STOPWORDS:
                            c["addr_num_idx"][(atok, num)].append(idx)

            count += 1
            if count % 1000000 == 0:
                print(f"  {count:,} records ({count/(time.time()-t0):.0f}/s)")

    print(f"  Done {filepath.name}: {count:,} in {time.time()-t0:.1f}s")
    return count

def make_countries():
    return defaultdict(lambda: {
        "ids": [], "name": [], "addr": [], "comp": [], "nums": [], "tokens": [],
        "comp_idx": defaultdict(list),
        "name_idx": defaultdict(list),
        "addr_num_idx": defaultdict(list),
    })

def run_precision_v11(test_dir, output_dir):
    print("=" * 70)
    print("Amazon ML Challenge 2026 - Pipeline V11 (Precision Strike)")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    match_out = output_dir / "matching_results.tsv"
    cand_out = output_dir / "candidate_pairs.tsv"

    countries = make_countries()
    total = 0
    for src in ["test_source2.tsv", "test_source3.tsv"]:
        total += build_index(test_dir / src, countries)

    print(f"\nIndexed total target records: {total:,}")

    s1_path = test_dir / "test_source1.tsv"
    n_s1 = n_matched = total_pairs = 0
    t0 = time.time()

    with open(s1_path, "r", encoding="utf-8") as fs, \
         open(match_out, "w", encoding="utf-8", newline="") as fm, \
         open(cand_out, "w", encoding="utf-8", newline="") as fc:

        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        next(fs)

        for line in fs:
            parts = line.split("\t")
            if not parts: continue
            sid = parts[0]
            name_raw = parts[1] if len(parts) > 1 else ""
            addr_raw = parts[2] if len(parts) > 2 else ""
            ctry = parts[3].strip() if len(parts) > 3 else "Unknown"

            c_data = countries.get(ctry)
            if not c_data:
                fm.write(f"{sid}\t\n")
                fc.write(f"{sid}\t\n")
                n_s1 += 1
                continue

            q_name = clean_text(name_raw)
            q_addr = clean_text(addr_raw)
            q_comp = compress_name(name_raw)
            q_tokens = get_tokens(q_name)
            q_nums = get_nums(q_addr)

            comp_idx = c_data["comp_idx"]
            name_idx = c_data["name_idx"]
            addr_num_idx = c_data["addr_num_idx"]

            cands = set()
            # 1. Exact compressed
            if q_comp and q_comp in comp_idx:
                cands.update(comp_idx[q_comp])
            # 2. Rarest name tokens
            if q_tokens:
                sorted_toks = sorted(q_tokens, key=lambda t: len(name_idx.get(t, ())))
                for t in sorted_toks[:2]:
                    postings = name_idx.get(t)
                    if postings and len(postings) <= MAX_NAME_POSTINGS:
                        cands.update(postings)
            # 3. Addr num
            if q_nums:
                for num in list(q_nums)[:2]:
                    for atok in q_addr.split()[:3]:
                        if len(atok) > 3 and atok not in STOPWORDS:
                            postings = addr_num_idx.get((atok, num))
                            if postings and len(postings) <= MAX_ADDR_POSTINGS:
                                cands.update(postings)

            if not cands:
                fm.write(f"{sid}\t\n")
                fc.write(f"{sid}\t\n")
                n_s1 += 1
                continue

            t_names = c_data["name"]
            t_addrs = c_data["addr"]
            t_comps = c_data["comp"]
            t_nums = c_data["nums"]
            t_ids = c_data["ids"]

            matches = []
            candidates_ranked = []

            for cid in cands:
                t_comp = t_comps[cid]
                t_n = t_names[cid]
                t_a = t_addrs[cid]
                t_num = t_nums[cid]

                # Hard Rejection: Conflicting Street / Building Numbers
                if q_nums and t_num and not (q_nums & t_num):
                    continue

                # Exact compressed match
                if q_comp and t_comp and q_comp == t_comp:
                    matches.append((1.0, t_ids[cid]))
                    candidates_ranked.append((1.0, t_ids[cid]))
                    continue

                nsim = fuzz.token_sort_ratio(q_name, t_n)
                asim = fuzz.token_sort_ratio(q_addr, t_a)
                num_match = bool(q_nums & t_num) if (q_nums and t_num) else False

                score = nsim * 0.6 + asim * 0.4

                # Precision Match Rules
                is_match = False
                if nsim >= 70 and (asim >= 40 or num_match or not t_a):
                    is_match = True
                elif nsim >= 55 and asim >= 75 and num_match:
                    is_match = True
                elif nsim >= 85:
                    is_match = True

                if is_match:
                    matches.append((score, t_ids[cid]))
                candidates_ranked.append((score, t_ids[cid]))

            matches.sort(key=lambda x: x[0], reverse=True)
            candidates_ranked.sort(key=lambda x: x[0], reverse=True)

            pred_ids = [eid for sc, eid in matches[:MAX_MATCHES]]
            cand_ids = [eid for sc, eid in candidates_ranked[:12]]

            fm.write(f"{sid}\t{','.join(pred_ids)}\n")
            fc.write(f"{sid}\t{','.join(cand_ids)}\n")

            n_s1 += 1
            if pred_ids:
                n_matched += 1
                total_pairs += len(pred_ids)

            if n_s1 % 50000 == 0:
                elapsed = time.time() - t0
                qps = n_s1 / elapsed
                rem = (1732544 - n_s1) / qps / 60
                fm.flush(); fc.flush()
                print(f"  [{n_s1:,}/1,732,544] {qps:.0f} qps | ETA {rem:.1f}m | "
                      f"matched: {n_matched:,} ({(1 - n_matched/n_s1)*100:.1f}% singletons) | "
                      f"avg matches: {total_pairs/max(1,n_matched):.2f}")

    singleton_pct = (n_s1 - n_matched) / n_s1 * 100
    print(f"\n{'='*70}")
    print("V11 PRECISION STRIKE COMPLETE!")
    print(f"  Total queries:    {n_s1:,}")
    print(f"  With matches:     {n_matched:,} ({n_matched/n_s1*100:.2f}%)")
    print(f"  Singletons:       {n_s1-n_matched:,} ({singleton_pct:.2f}%)")
    print(f"  Total pairs:      {total_pairs:,}")
    print(f"  Avg matches:      {total_pairs/max(1,n_matched):.2f}")
    print(f"  Total time:       {(time.time()-t0)/60:.2f} min")
    print(f"{'='*70}")

    zp = Path("submission.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(match_out, "output/matching_results.tsv")
        zf.write(cand_out, "output/candidate_pairs.tsv")
    print(f"Created submission.zip ({zp.stat().st_size/1024/1024:.1f} MB)")

if __name__ == "__main__":
    BASE = Path(".")
    TEST = BASE / "student_resource" / "dataset" / "test"
    OUT = BASE / "output"
    if TEST.exists():
        run_precision_v11(TEST, OUT)
    else:
        print("Test directory missing")
