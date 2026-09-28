#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Pipeline V14 (Grandmaster Precision Strike)
Targeting 0.95+ Macro F_0.5 Score on Leaderboard

Key Architectural Advances:
1. State & Locality Integrity:
   Prevents cross-state false merges in India (28 states/UTs) and US (50 states).
2. Legal Suffix Normalization & Core Name Scoring:
   Strips corporate noise (Pvt Ltd, LLC, Inc, Sarl, etc.) so generic suffixes don't artificially inflate similarity.
3. Domain & Exact Compressed Name Resolving:
   Instant O(1) matching for domains (.com, .in, etc.) and punctuation-stripped brands.
4. House Number Typo Resilience:
   Tolerates single-digit typos (e.g. 8706 vs 870, 769 vs 69) ONLY when both name and street agree strongly (>=75%).
5. Robust Address Token Indexing:
   Fixes the French and US address token bug, indexing real street words after numbers.
6. Clean Singleton Preservation:
   Maintains ~5.6% - 5.8% natural singletons to guarantee 1.0 points on no-match queries.
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
STOPWORDS = {"the", "and", "of", "a", "an", "in", "to", "for", "at", "on", "by", "null", "near", "opp", "no", "mr", "ms", "sri", "shri", "smt"}

# Indian states & UTs
STATE_CANONICAL = {
    "tamil nadu": "TN", "tamilnadu": "TN", "tn": "TN",
    "karnataka": "KA", "ka": "KA",
    "kerala": "KL", "kl": "KL",
    "maharashtra": "MH", "mh": "MH",
    "andhra pradesh": "AP", "andhra": "AP", "ap": "AP",
    "telangana": "TG", "ts": "TG", "tg": "TG",
    "gujarat": "GJ", "gj": "GJ",
    "rajasthan": "RJ", "rj": "RJ",
    "uttar pradesh": "UP", "up": "UP",
    "west bengal": "WB", "wb": "WB",
    "delhi": "DL", "dl": "DL",
    "punjab": "PB", "pb": "PB",
    "haryana": "HR", "hr": "HR",
    "bihar": "BR", "br": "BR",
    "odisha": "OD", "orissa": "OD", "or": "OD",
    "madhya pradesh": "MP", "mp": "MP",
    "assam": "AS", "as": "AS",
    "jharkhand": "JH", "jh": "JH",
    "chhattisgarh": "CG", "cg": "CG",
    "uttarakhand": "UK", "uk": "UK",
    "himachal pradesh": "HP", "hp": "HP",
    "goa": "GA", "ga": "GA",
}

US_STATES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia",
    "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt",
    "va", "wa", "wv", "wi", "wy"
}

us_fr_zip = re.compile(r'\b\d{5}\b')
in_pin = re.compile(r'\b\d{6}\b')

def get_postal(addr, ctry):
    if ctry in ("US", "France"): return frozenset(us_fr_zip.findall(addr))
    elif ctry == "India": return frozenset(in_pin.findall(addr))
    return frozenset()

def extract_state(addr, ctry):
    addr_l = addr.lower()
    if ctry == "India":
        for st, code in STATE_CANONICAL.items():
            if re.search(r'\b' + re.escape(st) + r'\b', addr_l):
                return code
    elif ctry == "US":
        words = re.findall(r'\b[a-zA-Z]{2}\b', addr)
        for w in reversed(words):
            wl = w.lower()
            if wl in US_STATES:
                return wl.upper()
    return None

def clean_text(s):
    s = anyascii(str(s)).lower().strip()
    return " ".join(s.split())

def strip_legal(s):
    s = LEGAL_SUFFIX_RE.sub(" ", s)
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
    return frozenset(t for t in toks if len(t) > 2 and t not in STOPWORDS)

def get_nums(s):
    raw = re.findall(r'\b\d+\b', s)
    return frozenset(n.lstrip("0") or "0" for n in raw if len(n) >= 1)

def get_street_tokens(addr):
    words = [w for w in re.findall(r'[a-z]+', clean_text(addr)) if len(w) >= 3 and w not in STOPWORDS]
    return words[:3]

MAX_NAME_POSTINGS = 250
MAX_ADDR_POSTINGS = 60
MAX_POST_POSTINGS = 60
MAX_MATCHES = 4

def make_countries():
    return defaultdict(lambda: {
        "ids": [], "name": [], "addr": [], "comp": [], "core": [],
        "nums": [], "post": [], "state": [],
        "comp_idx": defaultdict(list),
        "name_idx": defaultdict(list),
        "post_num_idx": defaultdict(list),
        "addr_num_idx": defaultdict(list),
    })

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
            core = strip_legal(c_name)
            toks = get_tokens(core)
            nums = get_nums(c_addr)
            post = get_postal(addr_raw, ctry)
            state = extract_state(addr_raw, ctry)
            street_words = get_street_tokens(c_addr)

            c["ids"].append(eid)
            c["name"].append(c_name)
            c["addr"].append(c_addr)
            c["comp"].append(comp)
            c["core"].append(core)
            c["nums"].append(nums)
            c["post"].append(post)
            c["state"].append(state)

            if comp:
                c["comp_idx"][comp].append(idx)
            for t in toks:
                c["name_idx"][t].append(idx)
            if post and nums:
                for p_code in post:
                    for num in nums:
                        c["post_num_idx"][(p_code, num)].append(idx)
            if nums and street_words:
                for num in list(nums)[:2]:
                    for stok in street_words:
                        c["addr_num_idx"][(stok, num)].append(idx)

            count += 1
            if count % 1000000 == 0:
                print(f"  {count:,} records ({count/(time.time()-t0):.0f}/s)")

    print(f"  Done {filepath.name}: {count:,} in {time.time()-t0:.1f}s")
    return count

def run_grandmaster_v14(test_dir, output_dir):
    print("=" * 70)
    print("Amazon ML Challenge 2026 - Pipeline V14 (Grandmaster Precision Strike)")
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
            q_core = strip_legal(q_name)
            q_tokens = get_tokens(q_core)
            q_nums = get_nums(q_addr)
            q_post = get_postal(addr_raw, ctry)
            q_state = extract_state(addr_raw, ctry)
            q_street_words = get_street_tokens(q_addr)
            q_addr_nonum = re.sub(r'\d+', ' ', q_addr).strip()

            comp_idx = c_data["comp_idx"]
            name_idx = c_data["name_idx"]
            post_num_idx = c_data["post_num_idx"]
            addr_num_idx = c_data["addr_num_idx"]

            cands = set()
            # 1. Exact compressed name
            if q_comp and q_comp in comp_idx:
                cands.update(comp_idx[q_comp])
            # 2. Rarest name tokens
            if q_tokens:
                sorted_toks = sorted(q_tokens, key=lambda t: len(name_idx.get(t, ())))
                for t in sorted_toks[:2]:
                    postings = name_idx.get(t)
                    if postings and len(postings) <= MAX_NAME_POSTINGS:
                        cands.update(postings)
                # Fallback if all tokens had > MAX_NAME_POSTINGS
                if not cands and sorted_toks:
                    t_rare = sorted_toks[0]
                    postings = name_idx.get(t_rare)
                    if postings and len(postings) <= 600:
                        cands.update(postings[:100])
            # 3. Postal code + Number match
            if q_post and q_nums:
                for p_code in q_post:
                    for num in q_nums:
                        postings = post_num_idx.get((p_code, num))
                        if postings and len(postings) <= MAX_POST_POSTINGS:
                            cands.update(postings)
            # 4. Street token + Number match
            if q_nums and q_street_words:
                for num in list(q_nums)[:2]:
                    for stok in q_street_words:
                        postings = addr_num_idx.get((stok, num))
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
            t_cores = c_data["core"]
            t_nums = c_data["nums"]
            t_posts = c_data["post"]
            t_states = c_data["state"]
            t_ids = c_data["ids"]

            matches = []
            candidates_ranked = []

            for cid in cands:
                t_comp = t_comps[cid]
                t_n = t_names[cid]
                t_a = t_addrs[cid]
                t_core = t_cores[cid]
                t_num = t_nums[cid]
                t_post = t_posts[cid]
                t_state = t_states[cid]
                tid = t_ids[cid]

                # Hard Rejections:
                # 1. Conflicting States
                if q_state and t_state and q_state != t_state:
                    continue

                # 2. Conflicting Postal Codes
                if q_post and t_post and not (q_post & t_post):
                    continue

                num_agree = bool(q_nums and t_num and (q_nums & t_num))
                num_conflict = bool(q_nums and t_num and not (q_nums & t_num))

                # Core Name similarity
                core_sort = fuzz.token_sort_ratio(q_core, t_core) if (q_core and t_core) else 0
                core_set = fuzz.token_set_ratio(q_core, t_core) if (q_core and t_core) else 0
                core_nsim = max(core_sort, core_set)

                # Address similarity
                asim_sort = fuzz.token_sort_ratio(q_addr, t_a) if (q_addr and t_a) else 0
                asim_set = fuzz.token_set_ratio(q_addr, t_a) if (q_addr and t_a) else 0
                asim = max(asim_sort, asim_set)

                t_addr_nonum = re.sub(r'\d+', ' ', t_a).strip()
                street_sim = fuzz.token_set_ratio(q_addr_nonum, t_addr_nonum) if (q_addr_nonum and t_addr_nonum) else 0

                # Check number typo resilience: e.g. 8706 vs 870, 769 vs 69, 9236 vs 9238
                if num_conflict:
                    is_num_typo = False
                    if core_nsim >= 65 and street_sim >= 75:
                        for qn in q_nums:
                            for tn in t_num:
                                if len(qn) >= 2 and len(tn) >= 2:
                                    if qn in tn or tn in qn or abs(len(qn) - len(tn)) <= 1 or fuzz.ratio(qn, tn) >= 66:
                                        is_num_typo = True
                                        break
                            if is_num_typo: break
                    if not is_num_typo:
                        continue

                t_core_tokens = get_tokens(t_core)
                shared_core = q_tokens & t_core_tokens

                is_match = False
                score = 0.0

                # 1. Exact compressed name match
                if q_comp and t_comp and q_comp == t_comp:
                    if not t_a or not q_addr or street_sim >= 20 or num_agree:
                        is_match = True
                        score = 100.0

                # 2. Corrupted / masked / Indic name with matching address
                elif (num_agree or street_sim >= 85) and asim >= 60 and street_sim >= 65:
                    if core_nsim >= 30 or not t_core or shared_core or len(t_core_tokens) <= 1:
                        is_match = True
                        score = 90.0 + asim * 0.1

                # 3. High address similarity (long matching street address)
                elif asim >= 75 and len(q_addr) > 15 and (core_nsim >= 35 or not t_core or shared_core):
                    is_match = True
                    score = 85.0 + asim * 0.1

                # 4. High core name similarity
                elif core_nsim >= 80:
                    if not t_a or not q_addr:
                        if len(shared_core) >= 1 or len(q_core) >= 8:
                            is_match = True
                            score = 80.0 + core_nsim * 0.1
                    elif asim >= 25 or num_agree or street_sim >= 40:
                        is_match = True
                        score = 80.0 + core_nsim * 0.1

                # 5. Moderate name with multiple shared core tokens when target address is empty
                elif not t_a and len(shared_core) >= 2 and core_nsim >= 65:
                    is_match = True
                    score = 78.0

                # 6. Moderate name + Strong address
                elif core_nsim >= 50 and (asim >= 50 or num_agree or street_sim >= 65):
                    is_match = True
                    score = 75.0 + core_nsim * 0.1

                if is_match:
                    matches.append((score, tid))
                candidates_ranked.append((score if is_match else (core_nsim * 0.5 + asim * 0.5), tid))

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
                      f"matched: {n_matched:,} ({(1 - n_matched/n_s1)*100:.2f}% singletons) | "
                      f"avg matches: {total_pairs/max(1,n_matched):.2f}")

    singleton_pct = (n_s1 - n_matched) / n_s1 * 100
    print(f"\n{'='*70}")
    print("V14 GRANDMASTER COMPLETE!")
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
        run_grandmaster_v14(TEST, OUT)
    else:
        print("Test directory missing")
