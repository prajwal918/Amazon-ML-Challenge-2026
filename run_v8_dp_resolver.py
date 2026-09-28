#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Production Pipeline V8 (Grandmaster DP Resolver)
Key Breakthroughs:
1. Pre-Vectorization Normalizer: De-leetspeaks ('5mart' -> 'smart'), accents, domains before indexing
2. 5-Key High-Recall Blocking: Exact compressed, 3 rarest tokens, addr+num, addr-only, prefix 4-gram
3. LightGBM Pairwise Probability Modeling (SIMD RapidFuzz features)
4. Faron / Ye et al. (ICML 2012) Expected F_0.5 Dynamic Programming:
   - Evaluates E[F_0.5] per query across k in {0, ..., K}
   - Automatically handles Singletons (k=0) when confidence is low
   - Stops at optimal k before false positives trigger the 4x precision penalty
"""
import sys, re, unicodedata, time, os, gc
from collections import defaultdict
from pathlib import Path
import numpy as np
import zipfile, shutil

try:
    from anyascii import anyascii
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
    import lightgbm as lgb
except ImportError:
    os.system("pip install -q anyascii rapidfuzz lightgbm")
    from anyascii import anyascii
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
    import lightgbm as lgb

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Pre-Vectorization Normalization ─────────────────────────────────────────
DOMAINS = (".com", ".net", ".org", ".co.in", ".in", ".fr", ".co", ".io", ".info", ".biz", ".org.in", ".gov.in")
LEGAL_SUFFIXES = {
    "corp", "corporation", "corporate", "pvt", "private", "ltd", "limited",
    "llc", "inc", "incorporated", "co", "company", "sarl", "sas", "llp", "sa", "gmbh",
    "sasu", "eurl", "dba", "services", "holdings", "group", "groupe", "center", "enterprises",
    "industries", "associates", "consulting", "solutions", "international", "intl"
}
STOPWORDS = {"the", "and", "of", "a", "an", "in", "to", "for", "at", "on", "by", "null", "near", "opp", "no", "mr", "ms", "sri", "shri", "smt"}
GENERIC_ADDR = {
    "road", "street", "avenue", "boulevard", "drive", "lane", "highway",
    "rd", "st", "ave", "dr", "ln", "hwy", "floor", "room",
    "unit", "building", "plot", "house", "city", "state",
    "india", "us", "usa", "france", "district", "block", "sector",
    "nagar", "colony", "marg", "gali", "ward", "taluk", "tehsil",
    "rue", "place", "allee", "chemin", "impasse", "route", "cours"
}

LEET_MAP = {'5':'s', '0':'o', '1':'i', '3':'e', '4':'a', '@':'a', '$':'s', '8':'b', '7':'t'}

def un_leetspeak(word):
    has_letters = any(c.isalpha() for c in word)
    has_digits = any(c.isdigit() or c in ('@', '$') for c in word)
    if has_letters and has_digits:
        if re.fullmatch(r'\d+[a-zA-Z]?', word):
            return word
        return "".join(LEET_MAP.get(c, c) for c in word)
    return word

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def clean_name(name):
    name = anyascii(str(name)).lower().strip()
    name = strip_accents(name)
    for ext in DOMAINS:
        if name.endswith(ext):
            name = name[:-len(ext)].strip()
    name = name.replace("&", " and ").replace("+", " and ")
    tokens = re.findall(r'[a-zA-Z0-9@$]+', name)
    cleaned = []
    for tok in tokens:
        tok = un_leetspeak(tok)
        if tok not in LEGAL_SUFFIXES and tok not in STOPWORDS and len(tok) > 1:
            cleaned.append(tok)
    return " ".join(cleaned)

def clean_addr(addr):
    addr = anyascii(str(addr)).lower().strip()
    addr = strip_accents(addr)
    addr = re.sub(r'\b0+(\d+)\b', r'\1', addr)  # strip leading zeros
    addr = re.sub(r'[^\w\s]', ' ', addr)
    return " ".join(addr.split())

def compress_name(n):
    n = clean_name(n)
    return re.sub(r'[^a-z0-9]', '', n)

def get_name_tokens(name_clean):
    return frozenset(t for t in name_clean.split() if len(t) > 1)

def get_nums(addr_clean):
    raw = re.findall(r'\b\d+\b', addr_clean)
    return frozenset(n.lstrip("0") or "0" for n in raw if len(n) >= 2)

def get_distinctive_addr_toks(addr_clean):
    return [t for t in addr_clean.split() if t not in STOPWORDS and t not in GENERIC_ADDR and len(t) > 2]

# ── Dynamic Programming: Expected F_0.5 Maximization ─────────────────────────
def expected_f05_optimal_k(probs, beta=0.5):
    """
    Exact Ye et al. (2012) / Faron DP optimizer for Macro F_0.5.
    Given sorted probabilities p_1 >= p_2 >= ... >= p_K:
    Finds k* in {0, ..., K} that maximizes E[F_0.5].
    Returns (k_optimal, best_expected_f).
    """
    if len(probs) == 0:
        return 0, 1.0
    probs = np.array(probs, dtype=np.float64)
    K = len(probs)
    beta_sq = beta ** 2    # 0.25
    coeff = 1.0 + beta_sq  # 1.25

    # Suffix sums of tail probabilities
    S = np.zeros(K + 1, dtype=np.float64)
    for i in range(K - 1, -1, -1):
        S[i] = S[i + 1] + probs[i]

    # Poisson binomial DP
    D = np.zeros((K + 1, K + 1), dtype=np.float64)
    D[0, 0] = 1.0
    for j in range(1, K + 1):
        p = probs[j - 1]
        D[j, 0] = D[j - 1, 0] * (1.0 - p)
        for m in range(1, j + 1):
            D[j, m] = D[j - 1, m] * (1.0 - p) + D[j - 1, m - 1] * p

    # k = 0 (singleton prediction)
    exp_f_0 = D[K, 0] * 1.0  # P(|Y|=0) * 1.0
    best_k = 0
    best_score = exp_f_0

    # k = 1, 2, ..., K
    for k in range(1, K + 1):
        tail_sum = S[k]
        exp_f_k = 0.0
        for m in range(1, k + 1):
            denom = beta_sq * (m + tail_sum) + k
            exp_f_k += D[k, m] * ((coeff * m) / denom)
        if exp_f_k > best_score:
            best_score = exp_f_k
            best_k = k

    return best_k, best_score

# ── Feature Extraction for LightGBM ──────────────────────────────────────────
def extract_features(s1_name, s1_addr, s1_comp, s1_nums, s1_tokens,
                     tgt_name, tgt_addr, tgt_comp, tgt_nums, tgt_tokens):
    jw = float(JaroWinkler.similarity(s1_name, tgt_name))
    tsort_name = fuzz.token_sort_ratio(s1_name, tgt_name) / 100.0
    tset_name = fuzz.token_set_ratio(s1_name, tgt_name) / 100.0
    tsort_addr = fuzz.token_sort_ratio(s1_addr, tgt_addr) / 100.0
    comp_match = 1.0 if (s1_comp and tgt_comp and s1_comp == tgt_comp) else 0.0
    num_match = 1.0 if (s1_nums and tgt_nums and bool(s1_nums & tgt_nums)) else 0.0
    
    n_inter = len(s1_tokens & tgt_tokens)
    n_union = len(s1_tokens | tgt_tokens)
    name_jacc = n_inter / n_union if n_union > 0 else 0.0
    len_diff = abs(len(s1_name) - len(tgt_name))

    return [jw, tsort_name, tset_name, tsort_addr, comp_match, num_match, name_jacc, len_diff]

# ── Index Construction ───────────────────────────────────────────────────────
MAX_NAME_POSTING = 300
MAX_ADDR_NUM_POSTING = 80
MAX_ADDR_ONLY_POSTING = 50
MAX_PREFIX_POSTING = 250

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
            
            c_name = clean_name(name_raw)
            c_addr = clean_addr(addr_raw)
            comp = compress_name(name_raw)
            ntoks = get_name_tokens(c_name)
            nums = get_nums(c_addr)
            dtoks = get_distinctive_addr_toks(c_addr)

            c["ids"].append(eid)
            c["name"].append(c_name)
            c["addr"].append(c_addr)
            c["comp"].append(comp)
            c["tokens"].append(ntoks)
            c["nums"].append(nums)

            # Key 1: Exact compressed
            if comp:
                c["comp_idx"][comp].append(idx)
            # Key 2: Name tokens
            for t in ntoks:
                c["name_idx"][t].append(idx)
            # Key 3: Addr + num
            if nums and dtoks:
                for num in list(nums)[:2]:
                    for atok in dtoks[:3]:
                        c["addr_num_idx"][(atok, num)].append(idx)
            # Key 4: Addr only
            if dtoks:
                for atok in dtoks[:2]:
                    c["addr_only_idx"][atok].append(idx)
            # Key 5: Compressed prefix 4-gram
            if comp and len(comp) >= 4:
                c["prefix_idx"][comp[:4]].append(idx)

            count += 1
            if count % 1000000 == 0:
                print(f"  {count:,} records ({count/(time.time()-t0):.0f}/s)")

    print(f"  Done {filepath.name}: {count:,} in {time.time()-t0:.1f}s")
    return count

def make_countries():
    return defaultdict(lambda: {
        "ids": [], "name": [], "addr": [], "comp": [], "tokens": [], "nums": [],
        "comp_idx": defaultdict(list),
        "name_idx": defaultdict(list),
        "addr_num_idx": defaultdict(list),
        "addr_only_idx": defaultdict(list),
        "prefix_idx": defaultdict(list),
    })

# ── Production Runner ────────────────────────────────────────────────────────
def run_production_v8(test_dir, model_path, output_dir):
    print("=" * 70)
    print("Amazon ML Challenge 2026 - Pipeline V8 (Grandmaster DP Resolver)")
    print("=" * 70)

    print(f"Loading LightGBM model from {model_path}...")
    model = lgb.Booster(model_file=str(model_path))
    print("Model loaded successfully!")

    output_dir.mkdir(parents=True, exist_ok=True)
    match_out = output_dir / "matching_results.tsv"
    cand_out = output_dir / "candidate_pairs.tsv"

    countries = make_countries()
    total = 0
    for src in ["test_source2.tsv", "test_source3.tsv"]:
        total += build_index(test_dir / src, countries)

    print(f"\nIndexed total target records: {total:,}")
    for ctry, c in sorted(countries.items()):
        print(f"  Country '{ctry}': {len(c['ids']):,} targets")

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

            q_name = clean_name(name_raw)
            q_addr = clean_addr(addr_raw)
            q_comp = compress_name(name_raw)
            q_tokens = get_name_tokens(q_name)
            q_nums = get_nums(q_addr)
            q_dtoks = get_distinctive_addr_toks(q_addr)

            comp_idx = c_data["comp_idx"]
            name_idx = c_data["name_idx"]
            addr_num_idx = c_data["addr_num_idx"]
            addr_only_idx = c_data["addr_only_idx"]
            prefix_idx = c_data["prefix_idx"]

            # ── BLOCKING: 5 Parallel Keys ──
            cands = set()
            # Key 1: Exact compressed
            if q_comp and q_comp in comp_idx:
                cands.update(comp_idx[q_comp])
            # Key 2: 3 Rarest name tokens
            if q_tokens:
                sorted_toks = sorted(q_tokens, key=lambda t: len(name_idx.get(t, ())))
                for t in sorted_toks[:3]:
                    postings = name_idx.get(t)
                    if postings and len(postings) <= MAX_NAME_POSTING:
                        cands.update(postings)
            # Key 3: Distinctive address + number
            if q_nums and q_dtoks:
                for num in list(q_nums)[:2]:
                    for atok in q_dtoks[:3]:
                        postings = addr_num_idx.get((atok, num))
                        if postings and len(postings) <= MAX_ADDR_NUM_POSTING:
                            cands.update(postings)
            # Key 4: Distinctive address only
            if q_dtoks and len(cands) < 150:
                for atok in q_dtoks[:2]:
                    postings = addr_only_idx.get(atok)
                    if postings and len(postings) <= MAX_ADDR_ONLY_POSTING:
                        cands.update(postings)
            # Key 5: Compressed prefix 4-gram
            if q_comp and len(q_comp) >= 4 and len(cands) < 150:
                postings = prefix_idx.get(q_comp[:4])
                if postings and len(postings) <= MAX_PREFIX_POSTING:
                    cands.update(postings)

            if not cands:
                fm.write(f"{sid}\t\n")
                fc.write(f"{sid}\t\n")
                n_s1 += 1
                continue

            # ── Pairwise Features & LightGBM Inference ──
            cand_list = list(cands)[:25]  # Evaluate up to top 25 candidates
            t_names = c_data["name"]
            t_addrs = c_data["addr"]
            t_comps = c_data["comp"]
            t_nums = c_data["nums"]
            t_tokens = c_data["tokens"]
            t_ids = c_data["ids"]

            feats = []
            for cid in cand_list:
                feat = extract_features(
                    q_name, q_addr, q_comp, q_nums, q_tokens,
                    t_names[cid], t_addrs[cid], t_comps[cid], t_nums[cid], t_tokens[cid]
                )
                feats.append(feat)

            probs = model.predict(np.array(feats, dtype=np.float32))

            # Sort candidates by descending probability
            sorted_pairs = sorted(zip(cand_list, probs), key=lambda x: x[1], reverse=True)
            sorted_cids = [cp[0] for cp in sorted_pairs]
            sorted_probs = [float(cp[1]) for cp in sorted_pairs]

            # ── Expected F_0.5 Dynamic Programming ──
            k_opt, best_exp_f = expected_f05_optimal_k(sorted_probs[:15])

            matched_ids = [t_ids[cid] for cid in sorted_cids[:k_opt]]
            cand_ids = [t_ids[cid] for cid in sorted_cids[:12]]

            fm.write(f"{sid}\t{','.join(matched_ids)}\n")
            fc.write(f"{sid}\t{','.join(cand_ids)}\n")

            n_s1 += 1
            if matched_ids:
                n_matched += 1
                total_pairs += len(matched_ids)

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
    print("V8 GRANDMASTER PIPELINE COMPLETE!")
    print(f"  Total queries:    {n_s1:,}")
    print(f"  With matches:     {n_matched:,} ({n_matched/n_s1*100:.2f}%)")
    print(f"  Singletons:       {n_s1-n_matched:,} ({singleton_pct:.2f}%)")
    print(f"  Total pairs:      {total_pairs:,}")
    print(f"  Avg matches:      {total_pairs/max(1,n_matched):.2f}")
    print(f"  Total time:       {(time.time()-t0)/60:.2f} min")
    print(f"{'='*70}")

    # Package
    zp = Path("submission.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(match_out, "output/matching_results.tsv")
        zf.write(cand_out, "output/candidate_pairs.tsv")
    print(f"Created submission.zip ({zp.stat().st_size/1024/1024:.1f} MB)")

    for d in [Path(r"~/Desktop"), Path(r"~/Desktop")]:
        if d.exists():
            shutil.copy2(match_out, d / "matching_results.tsv")
            shutil.copy2(zp, d / "submission.zip")
            print(f"Copied artifacts to {d}")

if __name__ == "__main__":
    BASE = Path(".")
    TEST = BASE / "student_resource" / "dataset" / "test"
    MODEL = BASE / "student_resource" / "models" / "lgb_entity_resolver.txt"
    OUT = BASE / "output"

    if TEST.exists() and MODEL.exists():
        run_production_v8(TEST, MODEL, OUT)
    else:
        print(f"Missing test data ({TEST.exists()}) or model ({MODEL.exists()})")
