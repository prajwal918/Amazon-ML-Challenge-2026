#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - V6 Maximum Recall + Precision Pipeline
Key improvements over V5:
1. 5 blocking keys (was 3) for maximum candidate recall
2. Address-only blocking WITHOUT requiring number match (catches different addresses)
3. Character 3-gram blocking on compressed names (catches truncated/partial names)
4. Higher posting caps (500 name, 100 addr)
5. fuzz.partial_ratio for truncated name matching ("B &" vs "B & G Exchange")
6. WRatio (weighted combo of all ratios) for robust scoring
7. Scores ALL candidates - no cap
"""
import sys, re, unicodedata, time, os, gc
from collections import defaultdict
from pathlib import Path
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

DOMAINS = (".com", ".net", ".org", ".co.in", ".in", ".fr", ".co", ".io", ".info", ".biz")
LEGAL = {
    "corp", "corporation", "corporate", "pvt", "private", "ltd", "limited",
    "llc", "inc", "incorporated", "co", "company", "sarl", "sas", "llp", "sa", "gmbh",
    "sasu", "eurl", "dba", "services", "holdings", "group", "groupe", "center", "enterprises"
}
STOP = {"the", "and", "of", "a", "an", "in", "to", "for", "at", "on", "by", "null", "near",
        "opp", "no", "mr", "ms", "sri", "shri", "smt"}
GENERIC_ADDR = {
    "road", "street", "avenue", "boulevard", "drive", "lane", "highway",
    "rd", "st", "ave", "dr", "ln", "hwy", "floor", "room",
    "unit", "building", "plot", "house", "city", "state",
    "india", "us", "usa", "france", "district", "block", "sector",
    "nagar", "colony", "marg", "gali", "ward", "taluk", "tehsil",
    "rue", "place", "allee", "chemin", "impasse", "route", "cours"
}

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def compress_name(n):
    n = strip_accents(str(n).lower().strip())
    for ext in DOMAINS:
        if n.endswith(ext): n = n[:-len(ext)]
    n = re.sub(r"[^a-z0-9]", "", n)
    for suf in ("corporation","corporate","private","limited","holdings","services",
                "enterprises","company","corp","pvt","ltd","llc","inc","gmbh",
                "sarl","sas","sasu","eurl","dba","group","groupe","center"):
        if n.endswith(suf): n = n[:-len(suf)]
    return n

def norm_name_for_fuzz(name):
    """Normalize for RapidFuzz comparison — anyascii for cross-script."""
    name = anyascii(str(name)).lower()
    name = re.sub(r"[^\w\s]", " ", name)
    toks = [t for t in name.split() if t not in LEGAL and t not in STOP and len(t) > 1]
    return " ".join(toks)

def norm_addr_for_fuzz(addr):
    addr = anyascii(str(addr)).lower()
    addr = re.sub(r'\b0+(\d)', r'\1', addr)  # strip leading zeros
    addr = re.sub(r"[^\w\s]", " ", addr)
    return " ".join(addr.split())

def get_name_tokens(name):
    name = strip_accents(str(name).lower())
    name = name.replace("&", " and ").replace("+", " and ")
    name = re.sub(r"[^\w\s]", " ", name)
    return frozenset(t for t in name.split() if t not in LEGAL and t not in STOP and len(t) > 1)

def get_nums(addr):
    raw = re.findall(r"\b\d+\b", str(addr))
    return frozenset(n.lstrip("0") or "0" for n in raw if len(n) >= 2)

def get_distinctive_addr_toks(addr):
    addr = strip_accents(str(addr).lower())
    addr = re.sub(r"[^\w\s]", " ", addr)
    return [t for t in addr.split() if t not in STOP and t not in GENERIC_ADDR and len(t) > 2]

def char_ngrams(s, n=3):
    if len(s) < n: return set()
    return {s[i:i+n] for i in range(len(s)-n+1)}

def is_latin(text):
    return all(ord(c) < 256 for c in text if c.isalpha())

# ── Index builder ────────────────────────────────────────────────────────
MAX_NAME_POSTING = 500
MAX_ADDR_NUM_POSTING = 100
MAX_ADDR_ONLY_POSTING = 60
MAX_NGRAM_POSTING = 300

def build_index(filepath, countries):
    print(f"Loading {filepath.name}...")
    t0 = time.time()
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.split("\t")
            if not p: continue
            eid = p[0]
            name = p[1] if len(p) > 1 else ""
            addr = p[2] if len(p) > 2 else ""
            ctry = p[3].strip() if len(p) > 3 else "Unknown"

            c = countries[ctry]
            idx = len(c["ids"])
            comp = compress_name(name)
            ntoks = get_name_tokens(name)
            nums = get_nums(addr)
            lat = is_latin(name)

            c["ids"].append(eid)
            c["name_raw"].append(name)
            c["addr_raw"].append(addr)
            c["comp"].append(comp)
            c["name_toks"].append(ntoks)
            c["nums"].append(nums)
            c["is_lat"].append(lat)

            # KEY 1: Compressed name exact
            if comp:
                c["comp_idx"][comp].append(idx)

            # KEY 2: Name token inverted index
            for t in ntoks:
                c["name_idx"][t].append(idx)

            # KEY 3: Distinctive address + number
            dtoks = get_distinctive_addr_toks(addr)
            if nums and dtoks:
                for num in list(nums)[:2]:
                    for atok in dtoks[:3]:
                        c["addr_num_idx"][(atok, num)].append(idx)

            # KEY 4: Address-only (distinctive token, no number required) — NEW
            if dtoks:
                for atok in dtoks[:2]:
                    c["addr_only_idx"][atok].append(idx)

            # KEY 5: Character 3-gram on compressed name — NEW
            if comp and len(comp) >= 4:
                prefix = comp[:5]
                c["ngram_idx"][prefix].append(idx)

            count += 1
            if count % 1000000 == 0:
                print(f"  {count:,} ({count/(time.time()-t0):.0f}/s)")

    print(f"  Done: {count:,} in {time.time()-t0:.1f}s")
    return count

def make_countries():
    return defaultdict(lambda: {
        "ids": [], "name_raw": [], "addr_raw": [],
        "comp": [], "name_toks": [], "nums": [], "is_lat": [],
        "comp_idx": defaultdict(list),
        "name_idx": defaultdict(list),
        "addr_num_idx": defaultdict(list),
        "addr_only_idx": defaultdict(list),
        "ngram_idx": defaultdict(list),
    })

# ── Matcher ──────────────────────────────────────────────────────────────
def match_query(name_raw, addr_raw, ctry, c_data):
    comp = compress_name(name_raw)
    ntoks = get_name_tokens(name_raw)
    nums = get_nums(addr_raw)
    nn = norm_name_for_fuzz(name_raw)
    na = norm_addr_for_fuzz(addr_raw)

    t_ids = c_data["ids"]
    t_name_raw = c_data["name_raw"]
    t_addr_raw = c_data["addr_raw"]
    t_comp = c_data["comp"]
    t_nums = c_data["nums"]
    t_lat = c_data["is_lat"]
    comp_idx = c_data["comp_idx"]
    name_idx = c_data["name_idx"]
    addr_num_idx = c_data["addr_num_idx"]
    addr_only_idx = c_data["addr_only_idx"]
    ngram_idx = c_data["ngram_idx"]

    # ── BLOCKING: 5 parallel keys ──
    cands = set()

    # Key 1: Compressed name exact
    if comp and comp in comp_idx:
        cands.update(comp_idx[comp])

    # Key 2: Rarest 4 name tokens
    if ntoks:
        sorted_t = sorted(ntoks, key=lambda t: len(name_idx.get(t, ())))
        for t in sorted_t[:4]:
            postings = name_idx.get(t)
            if postings and len(postings) <= MAX_NAME_POSTING:
                cands.update(postings)

    # Key 3: Distinctive address + number
    dtoks = get_distinctive_addr_toks(addr_raw)
    if nums and dtoks:
        for num in list(nums)[:2]:
            for atok in dtoks[:3]:
                postings = addr_num_idx.get((atok, num))
                if postings and len(postings) <= MAX_ADDR_NUM_POSTING:
                    cands.update(postings)

    # Key 4: Address-only (no number) — catches different address numbers
    if dtoks and len(cands) < 200:
        for atok in dtoks[:2]:
            postings = addr_only_idx.get(atok)
            if postings and len(postings) <= MAX_ADDR_ONLY_POSTING:
                cands.update(postings)

    # Key 5: Name prefix ngram — catches truncated/partial names
    if comp and len(comp) >= 4 and len(cands) < 200:
        prefix = comp[:5]
        postings = ngram_idx.get(prefix)
        if postings and len(postings) <= MAX_NGRAM_POSTING:
            cands.update(postings)

    if not cands:
        return [], []

    # ── SCORING: RapidFuzz on ALL candidates ──
    matches = []
    all_scored = []

    for cid in cands:
        tc = t_comp[cid]

        # Rule 1: Exact compressed name
        if comp and tc and comp == tc:
            matches.append((1.0, t_ids[cid]))
            all_scored.append((1.0, t_ids[cid]))
            continue

        t_nn = norm_name_for_fuzz(t_name_raw[cid])
        t_na = norm_addr_for_fuzz(t_addr_raw[cid])

        # RapidFuzz features (C++ SIMD — very fast)
        tsort_n = fuzz.token_sort_ratio(nn, t_nn)   # handles word permutations
        tset_n  = fuzz.token_set_ratio(nn, t_nn)     # handles subset names
        partial_n = fuzz.partial_ratio(nn, t_nn)      # handles truncated names
        tsort_a = fuzz.token_sort_ratio(na, t_na)     # address similarity
        has_num = bool(nums & t_nums[cid]) if nums and t_nums[cid] else False
        has_t_addr = bool(t_na.strip())
        t_is_lat = t_lat[cid]

        base_sc = tsort_n * 0.004 + tsort_a * 0.002
        all_scored.append((base_sc, t_ids[cid]))

        is_match = False
        sc = 0.0

        # Rule 2: Strong name similarity (handles permutations + typos)
        if tsort_n >= 65 and (tsort_a >= 35 or has_num or not has_t_addr):
            is_match = True
            sc = 0.90 + tsort_n / 100

        # Rule 3: Token set match (subset names: "B &" ↔ "B & G Exchange")
        elif tset_n >= 80 and tsort_n >= 45 and (tsort_a >= 30 or has_num):
            is_match = True
            sc = 0.85 + tset_n / 100

        # Rule 4: Partial match (truncated names: "Economic Development" ↔ "Economic Dvéegpemnt United Fellowship")
        elif partial_n >= 80 and tsort_n >= 40 and (tsort_a >= 35 or has_num):
            is_match = True
            sc = 0.82 + partial_n / 100

        # Rule 5: Moderate name + strong address
        elif tsort_n >= 50 and (tsort_a >= 55 or (has_num and tsort_a >= 35)):
            is_match = True
            sc = 0.80 + tsort_a / 100

        # Rule 6: Cross-script Indian (non-Latin target)
        elif not t_is_lat and ctry == "India" and (tsort_a >= 50 or (has_num and tsort_a >= 30)):
            is_match = True
            sc = 0.78 + tsort_a / 100

        # Rule 7: Strong address with number (parent/subsidiary at same location)
        elif has_num and tsort_a >= 60 and tsort_n >= 25:
            is_match = True
            sc = 0.72 + tsort_a / 100

        if is_match:
            matches.append((sc, t_ids[cid]))

    matches.sort(key=lambda x: x[0], reverse=True)
    all_scored.sort(key=lambda x: x[0], reverse=True)

    final_matches = [eid for _, eid in matches[:8]]
    cand_set = set(final_matches)
    final_cands = list(final_matches)
    for _, eid in all_scored:
        if eid not in cand_set:
            cand_set.add(eid)
            final_cands.append(eid)
        if len(final_cands) >= 12:
            break

    return final_matches, final_cands

# ── Evaluation helper ────────────────────────────────────────────────────
def compute_f05(pred_set, true_set):
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    tp = len(pred_set & true_set)
    if tp == 0: return 0.0
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return (1.25 * p * r) / (0.25 * p + r)

# ═══════════════════════════════════════════════════════════════════════════
# STAGE A: VALIDATION (if training data exists)
# ═══════════════════════════════════════════════════════════════════════════
def run_validation(train_dir, n_eval=15000):
    print("=" * 70)
    print(f"STAGE A: Validating on {n_eval:,} training queries (FULL S2+S3)")
    print("=" * 70)

    gt = {}
    with open(train_dir / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.strip().split("\t")
            gt[p[0]] = set(p[1].split(",")) if len(p) > 1 and p[1] else set()

    s1 = {}
    with open(train_dir / "train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i >= n_eval: break
            p = line.strip().split("\t")
            s1[p[0]] = (p[1] if len(p)>1 else "", p[2] if len(p)>2 else "", p[3].strip() if len(p)>3 else "Unknown")

    countries = make_countries()
    for src in ["train_source2.tsv", "train_source3.tsv"]:
        build_index(train_dir / src, countries)

    scores = []
    tot_tp = tot_fp = tot_fn = 0
    t0 = time.time()

    for i, (sid, (name, addr, ctry)) in enumerate(s1.items()):
        true_m = gt.get(sid, set())
        c_data = countries.get(ctry)
        preds = set()
        if c_data:
            matched, _ = match_query(name, addr, ctry, c_data)
            preds = set(matched)
        f05 = compute_f05(preds, true_m)
        scores.append(f05)
        tot_tp += len(preds & true_m)
        tot_fp += len(preds - true_m)
        tot_fn += len(true_m - preds)

        if (i+1) % 5000 == 0:
            m = sum(scores)/len(scores)
            p = tot_tp/(tot_tp+tot_fp) if (tot_tp+tot_fp)>0 else 0
            r = tot_tp/(tot_tp+tot_fn) if (tot_tp+tot_fn)>0 else 0
            print(f"  [{i+1:,}/{n_eval:,}] F05={m:.4f} P={p:.4f} R={r:.4f} ({(i+1)/(time.time()-t0):.0f} qps)")

    macro = sum(scores)/len(scores)
    p = tot_tp/(tot_tp+tot_fp) if (tot_tp+tot_fp)>0 else 0
    r = tot_tp/(tot_tp+tot_fn) if (tot_tp+tot_fn)>0 else 0
    print(f"\n  *** VALIDATION F_0.5 = {macro:.4f} (P={p:.4f} R={r:.4f}) ***\n")
    del countries; gc.collect()
    return macro

# ═══════════════════════════════════════════════════════════════════════════
# STAGE B: PRODUCTION
# ═══════════════════════════════════════════════════════════════════════════
def run_production(test_dir, output_dir):
    print("=" * 70)
    print("STAGE B: Production Run on Test Data")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    match_out = output_dir / "matching_results.tsv"
    cand_out = output_dir / "candidate_pairs.tsv"

    countries = make_countries()
    total = 0
    for src in ["test_source2.tsv", "test_source3.tsv"]:
        total += build_index(test_dir / src, countries)

    for ctry, c in sorted(countries.items()):
        print(f"  {ctry}: {len(c['ids']):,}")
    print(f"  TOTAL: {total:,}")

    n_s1 = n_matched = total_pairs = 0
    t0 = time.time()

    with open(test_dir / "test_source1.tsv", "r", encoding="utf-8") as fs, \
         open(match_out, "w", encoding="utf-8", newline="") as fm, \
         open(cand_out, "w", encoding="utf-8", newline="") as fc:

        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        next(fs)

        for line in fs:
            parts = line.split("\t")
            if not parts: continue
            sid = parts[0]
            name = parts[1] if len(parts)>1 else ""
            addr = parts[2] if len(parts)>2 else ""
            ctry = parts[3].strip() if len(parts)>3 else "Unknown"

            c_data = countries.get(ctry)
            if c_data:
                matched, cands = match_query(name, addr, ctry, c_data)
            else:
                matched, cands = [], []

            fm.write(f"{sid}\t{','.join(matched)}\n")
            fc.write(f"{sid}\t{','.join(cands)}\n")
            n_s1 += 1
            if matched:
                n_matched += 1
                total_pairs += len(matched)

            if n_s1 % 50000 == 0:
                elapsed = time.time() - t0
                qps = n_s1 / elapsed
                fm.flush(); fc.flush()
                print(f"  [{n_s1:,}/1,732,544] {qps:.0f} qps, ETA {(1732544-n_s1)/qps/60:.1f}m, "
                      f"matched:{n_matched:,}, avg:{total_pairs/max(1,n_matched):.2f}")

    singleton_pct = (n_s1 - n_matched) / n_s1 * 100
    print(f"\n{'='*70}")
    print(f"COMPLETE! queries={n_s1:,} matched={n_matched:,} singletons={n_s1-n_matched:,} ({singleton_pct:.1f}%)")
    print(f"pairs={total_pairs:,} avg={total_pairs/max(1,n_matched):.2f} time={int(time.time()-t0)}s")
    print(f"{'='*70}")

    # Package
    zp = Path("submission.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(match_out, "output/matching_results.tsv")
        zf.write(cand_out, "output/candidate_pairs.tsv")
    print(f"submission.zip: {zp.stat().st_size/1024/1024:.1f} MB")

    for d in [Path(r"~/Desktop"), Path(r"~/Desktop")]:
        if d.exists():
            shutil.copy2(match_out, d / "matching_results.tsv")
            shutil.copy2(zp, d / "submission.zip")
            print(f"Copied to {d}")

if __name__ == "__main__":
    BASE = Path(".")
    TRAIN = BASE / "student_resource" / "dataset" / "train"
    TEST = BASE / "student_resource" / "dataset" / "test"
    OUT = BASE / "output"

    if TRAIN.exists() and (TRAIN / "train_ground_truth.tsv").exists():
        run_validation(TRAIN, n_eval=15000)

    if TEST.exists():
        run_production(TEST, OUT)
    else:
        print("No test data found!")
