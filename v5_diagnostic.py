#!/usr/bin/env python3
"""
V5 Diagnostic: Run the FULL pipeline on 20,000 training queries against ALL
training S2+S3 targets (~8M records). This gives us a REALISTIC Macro F_0.5
score that should match the leaderboard. The goal is to understand WHERE the
pipeline is failing before deploying to the test set.
"""
import sys, re, unicodedata, time
from collections import defaultdict, Counter
from pathlib import Path
from anyascii import anyascii
from rapidfuzz import fuzz

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

TRAIN_DIR = Path("student_resource/dataset/train")

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
    """Normalize name for RapidFuzz comparison. Use anyascii for cross-script."""
    name = anyascii(str(name)).lower()
    name = re.sub(r"[^\w\s]", " ", name)
    toks = [t for t in name.split() if t not in LEGAL_SUFFIXES_SET and t not in STOPWORDS and len(t) > 1]
    return " ".join(toks)

def normalize_addr(addr):
    """Normalize address for comparison."""
    addr = anyascii(str(addr)).lower()
    # Strip leading zeros from numbers
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

def compute_f05(pred_set, true_set):
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return (1.25 * p * r) / (0.25 * p + r)

# ── Load Data ───────────────────────────────────────────────────────────────
print("=" * 70)
print("V5 DIAGNOSTIC: Full-Scale Training Validation")
print("=" * 70)

# Load ground truth (ALL of it)
print("Loading ALL ground truth...")
gt = {}
with open(TRAIN_DIR / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.strip().split("\t")
        gt[p[0]] = set(p[1].split(",")) if len(p) > 1 and p[1] else set()
print(f"  Ground truth loaded: {len(gt):,} entities")

# Load S1 queries (first 20,000)
N_EVAL = 20000
print(f"Loading first {N_EVAL:,} S1 queries...")
s1_data = {}
with open(TRAIN_DIR / "train_source1.tsv", "r", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= N_EVAL:
            break
        p = line.strip().split("\t")
        eid = p[0]
        name_raw = p[1] if len(p) > 1 else ""
        addr_raw = p[2] if len(p) > 2 else ""
        ctry = p[3].strip() if len(p) > 3 else "Unknown"
        s1_data[eid] = (name_raw, addr_raw, ctry)

print(f"  Loaded {len(s1_data):,} S1 queries")

# Load ALL S2+S3 targets into per-country arrays
print("Loading ALL training S2+S3 targets...")
countries = defaultdict(lambda: {
    "ids": [], "name_raw": [], "addr_raw": [],
    "comp": [], "name_toks": [], "nums": [],
    "is_lat": [],
    "comp_idx": defaultdict(list),
    "name_idx": defaultdict(list),
    "addr_num_idx": defaultdict(list),
})

total_targets = 0
for src in ["train_source2.tsv", "train_source3.tsv"]:
    t0 = time.time()
    count = 0
    with open(TRAIN_DIR / src, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.split("\t")
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
                print(f"    {src}: {count:,} loaded ({count/(time.time()-t0):.0f}/s)")

    total_targets += count
    print(f"  {src}: {count:,} records in {time.time()-t0:.1f}s")

print(f"\n  TOTAL targets indexed: {total_targets:,}")
for ctry, c in sorted(countries.items()):
    print(f"    {ctry}: {len(c['ids']):,} targets, {len(c['comp_idx']):,} unique names")

# ── Matching Engine ─────────────────────────────────────────────────────────
MAX_POSTING = 200
MAX_ADDR_POSTING = 40

def match_query(name_raw, addr_raw, ctry, c_data):
    """Returns list of (score, target_eid) for accepted matches."""
    comp = compress_name(name_raw)
    ntoks = get_name_tokens(name_raw)
    nums = get_nums(addr_raw)
    norm_name = normalize_name(name_raw)
    norm_addr = normalize_addr(addr_raw)

    target_ids = c_data["ids"]
    target_name_raw = c_data["name_raw"]
    target_addr_raw = c_data["addr_raw"]
    target_comp = c_data["comp"]
    target_ntoks = c_data["name_toks"]
    target_nums = c_data["nums"]
    target_lat = c_data["is_lat"]
    comp_idx = c_data["comp_idx"]
    name_idx = c_data["name_idx"]
    addr_num_idx = c_data["addr_num_idx"]

    # ── Stage 1: Blocking ──
    cands = set()
    # Key 1: Compressed name
    if comp and comp in comp_idx:
        cands.update(comp_idx[comp])
    # Key 2: Rarest 2 name tokens
    if ntoks:
        sorted_toks = sorted(ntoks, key=lambda t: len(name_idx.get(t, ())))
        for t in sorted_toks[:3]:
            postings = name_idx.get(t)
            if postings:
                if len(postings) <= MAX_POSTING:
                    cands.update(postings)
                elif len(cands) < 5:
                    cands.update(postings[:MAX_POSTING])
    # Key 3: Distinctive address + number
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
        return []

    # ── Stage 2: RapidFuzz Scoring ──
    matches = []
    for cid in cands:
        t_comp = target_comp[cid]
        t_lat = target_lat[cid]

        # Rule 1: Exact compressed name match
        if comp and t_comp and comp == t_comp:
            matches.append((1.0, target_ids[cid]))
            continue

        # Compute RapidFuzz features
        t_norm_name = normalize_name(target_name_raw[cid])
        t_norm_addr = normalize_addr(target_addr_raw[cid])
        
        tsort_name = fuzz.token_sort_ratio(norm_name, t_norm_name)
        tset_name = fuzz.token_set_ratio(norm_name, t_norm_name)
        tsort_addr = fuzz.token_sort_ratio(norm_addr, t_norm_addr)
        has_num = bool(nums & target_nums[cid]) if nums and target_nums[cid] else False
        has_t_addr = bool(t_norm_addr.strip())

        # Rule 2: High name similarity + address support
        if tsort_name >= 70 and (tsort_addr >= 40 or has_num or not has_t_addr):
            sc = 0.90 + tsort_name / 100 + tsort_addr / 200
            matches.append((sc, target_ids[cid]))
        # Rule 3: Token set match (handles subset names like "B &" vs "B & G Exchange")
        elif tset_name >= 85 and tsort_name >= 50 and (tsort_addr >= 35 or has_num):
            sc = 0.85 + tset_name / 100 + tsort_addr / 200
            matches.append((sc, target_ids[cid]))
        # Rule 4: Moderate name + strong address
        elif tsort_name >= 55 and (tsort_addr >= 55 or (has_num and tsort_addr >= 35)):
            sc = 0.80 + tsort_addr / 100
            matches.append((sc, target_ids[cid]))
        # Rule 5: Cross-script Indian matching (non-Latin target)
        elif not t_lat and ctry == "India" and (tsort_addr >= 55 or (has_num and tsort_addr >= 35)):
            sc = 0.78 + tsort_addr / 100
            matches.append((sc, target_ids[cid]))
        # Rule 6: Strong address match with number
        elif has_num and tsort_addr >= 65:
            sc = 0.72 + tsort_addr / 100
            matches.append((sc, target_ids[cid]))

    matches.sort(key=lambda x: x[0], reverse=True)
    return matches[:8]  # Max 8 matches

# ── Run Evaluation ──────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"Running V5 pipeline on {len(s1_data):,} training queries...")
print(f"{'=' * 70}")

scores = []
tot_tp = tot_fp = tot_fn = 0
n_singleton_pred = 0
n_singleton_true = 0
n_zero_f05 = 0
error_examples = []
t0 = time.time()
processed = 0

for s1_id, (name_raw, addr_raw, ctry) in s1_data.items():
    true_m = gt.get(s1_id, set())
    c_data = countries.get(ctry)

    if not c_data:
        preds = set()
    else:
        matched = match_query(name_raw, addr_raw, ctry, c_data)
        preds = {eid for sc, eid in matched}

    f05 = compute_f05(preds, true_m)
    scores.append(f05)
    
    tp = len(preds & true_m)
    fp = len(preds - true_m)
    fn = len(true_m - preds)
    tot_tp += tp
    tot_fp += fp
    tot_fn += fn
    
    if not preds:
        n_singleton_pred += 1
    if not true_m:
        n_singleton_true += 1
    if f05 == 0.0 and true_m:
        n_zero_f05 += 1
        if len(error_examples) < 10:
            error_examples.append((s1_id, name_raw, addr_raw, ctry, true_m, preds))

    processed += 1
    if processed % 5000 == 0:
        elapsed = time.time() - t0
        qps = processed / elapsed
        macro = sum(scores) / len(scores)
        p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0
        r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0
        print(f"  [{processed:,} / {len(s1_data):,}] "
              f"Macro F_0.5 = {macro:.4f} | P = {p:.4f} | R = {r:.4f} | "
              f"{qps:.0f} qps | ETA: {(len(s1_data)-processed)/qps/60:.1f} min")

macro_f05 = sum(scores) / len(scores)
prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0
rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0

print(f"\n{'=' * 70}")
print(f"V5 FULL-SCALE VALIDATION RESULTS ({len(scores):,} queries)")
print(f"{'=' * 70}")
print(f"  Macro F_0.5:          {macro_f05:.4f}")
print(f"  Overall Precision:    {prec:.4f} ({tot_tp:,} TP / {tot_fp:,} FP)")
print(f"  Overall Recall:       {rec:.4f} ({tot_tp:,} TP / {tot_fn:,} FN)")
print(f"  Predicted Singletons: {n_singleton_pred:,} ({n_singleton_pred/len(scores)*100:.2f}%)")
print(f"  True Singletons:      {n_singleton_true:,} ({n_singleton_true/len(scores)*100:.2f}%)")
print(f"  Entities with F=0.0:  {n_zero_f05:,} ({n_zero_f05/len(scores)*100:.2f}%)")
print(f"  Total query time:     {time.time()-t0:.1f}s ({processed/(time.time()-t0):.0f} qps)")

if error_examples:
    print(f"\n  TOP ERROR EXAMPLES (F_0.5 = 0.0):")
    for s1_id, name, addr, ctry, true_m, preds in error_examples[:5]:
        print(f"    [{s1_id}] ({ctry}) '{name}' | '{addr}'")
        print(f"      True matches: {true_m}")
        print(f"      Predicted:    {preds}")
        print()
