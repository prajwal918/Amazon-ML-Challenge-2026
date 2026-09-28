#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - V5 Combined Pipeline
Stage A: Validate on training data to confirm score
Stage B: Run on test data for submission
All on Kaggle Cloud (30 GB RAM)
"""
import sys, re, unicodedata, time, os
from collections import defaultdict, Counter
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from anyascii import anyascii
    from rapidfuzz import fuzz
except ImportError:
    os.system("pip install -q anyascii rapidfuzz")
    from anyascii import anyascii
    from rapidfuzz import fuzz

# ── Normalization ───────────────────────────────────────────────────────────
DOMAINS = (".com", ".net", ".org", ".co.in", ".in", ".fr", ".co", ".io", ".info", ".biz")
LEGAL_SUFFIXES_SET = {
    "corp", "corporation", "corporate", "pvt", "private", "ltd", "limited",
    "llc", "inc", "incorporated", "co", "company", "sarl", "sas", "llp", "sa", "gmbh",
    "sasu", "eurl", "dba", "services", "holdings", "group", "groupe", "center", "enterprises"
}
STOPWORDS = {"the", "and", "of", "a", "an", "in", "to", "for", "at", "on", "by", "null"}
GENERIC_ADDR = {
    "road", "street", "avenue", "boulevard", "drive", "lane", "highway",
    "rd", "st", "ave", "dr", "ln", "hwy", "near", "opp", "floor", "room",
    "unit", "building", "plot", "no", "house", "city", "state", "india", "us", "france"
}

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def compress_name(n):
    n = strip_accents(str(n).lower().strip())
    for ext in DOMAINS:
        if n.endswith(ext):
            n = n[:-len(ext)]
    n = re.sub(r"[^a-z0-9]", "", n)
    for suf in ("corporation", "corporate", "private", "limited", "holdings", "services",
                "enterprises", "company", "corp", "pvt", "ltd", "llc", "inc", "gmbh",
                "sarl", "sas", "sasu", "eurl", "dba", "group", "groupe", "center"):
        if n.endswith(suf):
            n = n[:-len(suf)]
    return n

def normalize_name(name):
    name = anyascii(str(name)).lower()
    name = re.sub(r"[^\w\s]", " ", name)
    toks = [t for t in name.split() if t not in LEGAL_SUFFIXES_SET and t not in STOPWORDS and len(t) > 1]
    return " ".join(toks)

def normalize_addr(addr):
    addr = anyascii(str(addr)).lower()
    addr = re.sub(r'\b0+(\d)', r'\1', addr)
    addr = re.sub(r"[^\w\s]", " ", addr)
    return " ".join(addr.split())

def get_name_tokens(name):
    name = strip_accents(str(name).lower())
    name = name.replace("&", " and ").replace("+", " and ")
    name = re.sub(r"[^\w\s]", " ", name)
    return frozenset(t for t in name.split() if t not in LEGAL_SUFFIXES_SET and t not in STOPWORDS and len(t) > 1)

def get_nums(addr):
    raw = re.findall(r"\b\d+\b", str(addr))
    return frozenset(n.lstrip("0") or "0" for n in raw if len(n) >= 2)

def get_distinctive_addr_toks(addr):
    addr = strip_accents(str(addr).lower())
    addr = re.sub(r"[^\w\s]", " ", addr)
    return [t for t in addr.split() if t not in STOPWORDS and t not in GENERIC_ADDR and len(t) > 2]

def is_latin(text):
    return all(ord(c) < 256 for c in text if c.isalpha())

MAX_POSTING = 200
MAX_ADDR_POSTING = 40

def build_index(filepath, countries):
    print(f"Loading and indexing {filepath.name}...")
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

            if comp:
                c["comp_idx"][comp].append(idx)
            for t in ntoks:
                c["name_idx"][t].append(idx)
            if nums:
                dtoks = get_distinctive_addr_toks(addr)
                for num in list(nums)[:2]:
                    for atok in dtoks[:3]:
                        c["addr_num_idx"][(atok, num)].append(idx)

            count += 1
            if count % 1000000 == 0:
                print(f"  ... {count:,} records ({count/(time.time()-t0):.0f}/s)")

    print(f"  Finished {filepath.name}: {count:,} records in {time.time()-t0:.1f}s")
    return count

def match_query(name_raw, addr_raw, ctry, c_data):
    comp = compress_name(name_raw)
    ntoks = get_name_tokens(name_raw)
    nums = get_nums(addr_raw)
    norm_name = normalize_name(name_raw)
    norm_addr = normalize_addr(addr_raw)

    comp_idx = c_data["comp_idx"]
    name_idx = c_data["name_idx"]
    addr_num_idx = c_data["addr_num_idx"]
    t_ids = c_data["ids"]
    t_name_raw = c_data["name_raw"]
    t_addr_raw = c_data["addr_raw"]
    t_comp = c_data["comp"]
    t_nums = c_data["nums"]
    t_lat = c_data["is_lat"]

    # ── Blocking ──
    cands = set()
    if comp and comp in comp_idx:
        cands.update(comp_idx[comp])
    if ntoks:
        sorted_toks = sorted(ntoks, key=lambda t: len(name_idx.get(t, ())))
        for t in sorted_toks[:3]:
            postings = name_idx.get(t)
            if postings:
                if len(postings) <= MAX_POSTING:
                    cands.update(postings)
                elif len(cands) < 5:
                    cands.update(postings[:MAX_POSTING])
    if nums:
        dtoks = get_distinctive_addr_toks(addr_raw)
        for num in list(nums)[:2]:
            for atok in dtoks[:3]:
                postings = addr_num_idx.get((atok, num))
                if postings:
                    if len(postings) <= MAX_ADDR_POSTING:
                        cands.update(postings)
                    elif len(cands) < 5:
                        cands.update(postings[:MAX_ADDR_POSTING])

    if not cands:
        return [], []

    # ── Scoring (ALL candidates, no cap) ──
    matches = []
    all_cands_scored = []

    for cid in cands:
        tc = t_comp[cid]

        # Rule 1: Exact compressed name
        if comp and tc and comp == tc:
            matches.append((1.0, t_ids[cid]))
            all_cands_scored.append((1.0, t_ids[cid]))
            continue

        t_nn = normalize_name(t_name_raw[cid])
        t_na = normalize_addr(t_addr_raw[cid])

        tsort_n = fuzz.token_sort_ratio(norm_name, t_nn)
        tset_n = fuzz.token_set_ratio(norm_name, t_nn)
        tsort_a = fuzz.token_sort_ratio(norm_addr, t_na)
        has_num = bool(nums & t_nums[cid]) if nums and t_nums[cid] else False
        has_t_addr = bool(t_na.strip())

        sc = tsort_n * 0.004 + tsort_a * 0.002  # base score for candidate ranking
        all_cands_scored.append((sc, t_ids[cid]))

        is_match = False

        # Rule 2: Strong name match (token_sort handles permutations + typos)
        if tsort_n >= 70 and (tsort_a >= 40 or has_num or not has_t_addr):
            is_match = True
            sc = 0.90 + tsort_n / 100
        # Rule 3: Token set match (subset names: "B &" vs "B & G Exchange")
        elif tset_n >= 85 and tsort_n >= 50 and (tsort_a >= 35 or has_num):
            is_match = True
            sc = 0.85 + tset_n / 100
        # Rule 4: Moderate name + strong address
        elif tsort_n >= 55 and (tsort_a >= 55 or (has_num and tsort_a >= 35)):
            is_match = True
            sc = 0.80 + tsort_a / 100
        # Rule 5: Cross-script Indian (non-Latin target name)
        elif not t_lat[cid] and ctry == "India" and (tsort_a >= 55 or (has_num and tsort_a >= 35)):
            is_match = True
            sc = 0.78 + tsort_a / 100
        # Rule 6: Strong address with number overlap
        elif has_num and tsort_a >= 65:
            is_match = True
            sc = 0.72 + tsort_a / 100

        if is_match:
            matches.append((sc, t_ids[cid]))

    matches.sort(key=lambda x: x[0], reverse=True)
    all_cands_scored.sort(key=lambda x: x[0], reverse=True)

    final_matches = [eid for _, eid in matches[:8]]
    cand_set = set(final_matches)
    final_cands = list(final_matches)
    for _, eid in all_cands_scored:
        if eid not in cand_set:
            cand_set.add(eid)
            final_cands.append(eid)
        if len(final_cands) >= 12:
            break

    return final_matches, final_cands

def make_countries():
    return defaultdict(lambda: {
        "ids": [], "name_raw": [], "addr_raw": [],
        "comp": [], "name_toks": [], "nums": [], "is_lat": [],
        "comp_idx": defaultdict(list),
        "name_idx": defaultdict(list),
        "addr_num_idx": defaultdict(list),
    })

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
# STAGE A: VALIDATION ON TRAINING DATA
# ═══════════════════════════════════════════════════════════════════════════
def run_validation(train_dir, n_eval=20000):
    print("=" * 70)
    print(f"STAGE A: Validating on {n_eval:,} training queries (FULL target space)")
    print("=" * 70)

    gt = {}
    with open(train_dir / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.strip().split("\t")
            gt[p[0]] = set(p[1].split(",")) if len(p) > 1 and p[1] else set()

    s1_data = {}
    with open(train_dir / "train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i >= n_eval: break
            p = line.strip().split("\t")
            s1_data[p[0]] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3].strip() if len(p) > 3 else "Unknown")

    countries = make_countries()
    for src in ["train_source2.tsv", "train_source3.tsv"]:
        build_index(train_dir / src, countries)

    scores = []
    tot_tp = tot_fp = tot_fn = 0
    t0 = time.time()

    for i, (s1_id, (name, addr, ctry)) in enumerate(s1_data.items()):
        true_m = gt.get(s1_id, set())
        c_data = countries.get(ctry)
        if c_data:
            matched, _ = match_query(name, addr, ctry, c_data)
            preds = set(matched)
        else:
            preds = set()

        f05 = compute_f05(preds, true_m)
        scores.append(f05)
        tot_tp += len(preds & true_m)
        tot_fp += len(preds - true_m)
        tot_fn += len(true_m - preds)

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            m = sum(scores) / len(scores)
            p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0
            r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0
            print(f"  [{i+1:,}/{n_eval:,}] F_0.5={m:.4f} P={p:.4f} R={r:.4f} ({(i+1)/elapsed:.0f} qps)")

    macro = sum(scores) / len(scores)
    p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0
    r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0
    print(f"\n  VALIDATION MACRO F_0.5: {macro:.4f}  (P={p:.4f}, R={r:.4f})")
    print(f"  Time: {time.time()-t0:.1f}s\n")

    # Free memory
    del countries, s1_data, gt
    import gc; gc.collect()
    return macro

# ═══════════════════════════════════════════════════════════════════════════
# STAGE B: PRODUCTION RUN ON TEST DATA
# ═══════════════════════════════════════════════════════════════════════════
def run_production(test_dir, output_dir):
    import zipfile, shutil

    print("=" * 70)
    print("STAGE B: Production Run on Test Data")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    match_out = output_dir / "matching_results.tsv"
    cand_out = output_dir / "candidate_pairs.tsv"

    countries = make_countries()
    for src in ["test_source2.tsv", "test_source3.tsv"]:
        build_index(test_dir / src, countries)

    print(f"\nTotal targets indexed:")
    total = 0
    for ctry, c in sorted(countries.items()):
        print(f"  {ctry}: {len(c['ids']):,}")
        total += len(c['ids'])
    print(f"  TOTAL: {total:,}")

    s1_path = test_dir / "test_source1.tsv"
    n_s1 = 0
    n_matched = 0
    total_pairs = 0
    t0 = time.time()

    with open(s1_path, "r", encoding="utf-8") as f_s1, \
         open(match_out, "w", encoding="utf-8", newline="") as f_m, \
         open(cand_out, "w", encoding="utf-8", newline="") as f_c:

        f_m.write("source1_entity_id\tmatched_entity_ids\n")
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")
        next(f_s1)

        for line in f_s1:
            parts = line.split("\t")
            if not parts: continue
            s1_id = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            ctry = parts[3].strip() if len(parts) > 3 else "Unknown"

            c_data = countries.get(ctry)
            if c_data:
                matched, cands = match_query(name, addr, ctry, c_data)
            else:
                matched, cands = [], []

            f_m.write(f"{s1_id}\t{','.join(matched)}\n")
            f_c.write(f"{s1_id}\t{','.join(cands)}\n")

            n_s1 += 1
            if matched:
                n_matched += 1
                total_pairs += len(matched)

            if n_s1 % 50000 == 0:
                elapsed = time.time() - t0
                qps = n_s1 / elapsed
                rem = (1732544 - n_s1) / qps / 60
                f_m.flush()
                f_c.flush()
                print(f"  [{n_s1:,}/1,732,544] {qps:.0f} qps, ETA {rem:.1f} min, "
                      f"matches: {n_matched:,}, avg: {total_pairs/max(1,n_matched):.2f}")

    print(f"\n{'=' * 70}")
    print("PRODUCTION PIPELINE COMPLETE!")
    print(f"  Total queries:    {n_s1:,}")
    print(f"  With matches:     {n_matched:,} ({n_matched/n_s1*100:.2f}%)")
    print(f"  Singletons:       {n_s1-n_matched:,} ({(n_s1-n_matched)/n_s1*100:.2f}%)")
    print(f"  Total pairs:      {total_pairs:,}")
    print(f"  Avg matches:      {total_pairs/max(1,n_matched):.2f}")
    print(f"  Time:             {(time.time()-t0)/60:.2f} min")
    print(f"{'=' * 70}")

    # Package
    zip_path = Path(".") / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(match_out, "output/matching_results.tsv")
        zf.write(cand_out, "output/candidate_pairs.tsv")
    print(f"Created {zip_path} ({zip_path.stat().st_size/1024/1024:.1f} MB)")

    # Copy to desktop if local
    for d in [Path(r"~/Desktop"), Path(r"~/Desktop")]:
        if d.exists():
            shutil.copy2(match_out, d / "matching_results.tsv")
            shutil.copy2(zip_path, d / "submission.zip")
            print(f"Copied to {d}")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    BASE = Path(".")
    TRAIN_DIR = BASE / "student_resource" / "dataset" / "train"
    TEST_DIR = BASE / "student_resource" / "dataset" / "test"
    OUTPUT_DIR = BASE / "output"

    # If training data exists, validate first
    if TRAIN_DIR.exists() and (TRAIN_DIR / "train_ground_truth.tsv").exists():
        val_score = run_validation(TRAIN_DIR, n_eval=20000)
        print(f"\n*** VALIDATION SCORE: {val_score:.4f} ***\n")
    else:
        print("No training data found, skipping validation.\n")

    # Run production on test data
    if TEST_DIR.exists():
        run_production(TEST_DIR, OUTPUT_DIR)
    else:
        print("ERROR: Test data not found!")
