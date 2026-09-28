import sys, re, time
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np

from anyascii import anyascii
from rapidfuzz import fuzz

DATA_DIR = Path("student_resource/dataset/train")

print("Loading ground truth...")
gt_df = pd.read_csv(DATA_DIR / "train_ground_truth.tsv", sep="\t")
gt_dict = {}
all_matched_targets = set()
for _, row in gt_df.iterrows():
    qid = str(row['source1_entity_id'])
    m = str(row['matched_entity_ids'])
    if pd.isna(row['matched_entity_ids']) or not m or m == "nan":
        gt_dict[qid] = []
    else:
        targets = [x.strip() for x in m.split(",") if x.strip()]
        gt_dict[qid] = targets
        all_matched_targets.update(targets)

print(f"Total ground truth records: {len(gt_dict):,}")
singletons = sum(1 for v in gt_dict.values() if len(v) == 0)
print(f"Total singletons: {singletons:,} ({singletons/len(gt_dict)*100:.2f}%)")

# Sample 3000 queries: 2700 with matches, 300 singletons
sample_qids = []
s_count = 0
m_count = 0
for qid, targets in gt_dict.items():
    if len(targets) == 0 and s_count < 300:
        sample_qids.append(qid)
        s_count += 1
    elif len(targets) > 0 and m_count < 2700:
        sample_qids.append(qid)
        m_count += 1
    if s_count >= 300 and m_count >= 2700:
        break

sample_qids_set = set(sample_qids)
sample_targets = set()
for qid in sample_qids:
    sample_targets.update(gt_dict[qid])

print(f"Sampled {len(sample_qids)} queries ({s_count} singletons, {m_count} with matches)")
print(f"True targets to find: {len(sample_targets)}")

# Load S1 for sampled queries
s1_records = {}
print("Loading train_source1 for sample...")
with open(DATA_DIR / "train_source1.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("\t")
        if p[0] in sample_qids_set:
            s1_records[p[0]] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else "Unknown")
            if len(s1_records) == len(sample_qids):
                break

print(f"Loaded {len(s1_records)} query records.")

# Index targets: sample_targets + 50,000 random distractors from S2 and S3
print("Loading target records from S2 and S3...")
target_records = {}
for src_file in ["train_source2.tsv", "train_source3.tsv"]:
    distractors_loaded = 0
    with open(DATA_DIR / src_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            eid = p[0]
            if eid in sample_targets:
                target_records[eid] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else "Unknown")
            elif distractors_loaded < 25000:
                target_records[eid] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else "Unknown")
                distractors_loaded += 1

print(f"Total target records indexed: {len(target_records):,} (includes {len(sample_targets & set(target_records.keys()))} true matches)")

# Define V11 rules to evaluate
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

# Build index for target records
countries = defaultdict(lambda: {
    "ids": [], "name": [], "addr": [], "comp": [], "nums": [], "tokens": [],
    "comp_idx": defaultdict(list),
    "name_idx": defaultdict(list),
    "addr_num_idx": defaultdict(list),
})

for eid, (name, addr, ctry) in target_records.items():
    c = countries[ctry]
    idx = len(c["ids"])
    c_name = clean_text(name)
    c_addr = clean_text(addr)
    comp = compress_name(name)
    toks = get_tokens(c_name)
    nums = get_nums(c_addr)

    c["ids"].append(eid)
    c["name"].append(c_name)
    c["addr"].append(c_addr)
    c["comp"].append(comp)
    c["nums"].append(nums)
    c["tokens"].append(toks)

    if comp: c["comp_idx"][comp].append(idx)
    for t in toks: c["name_idx"][t].append(idx)
    if nums:
        for num in list(nums)[:2]:
            for atok in c_addr.split()[:3]:
                if len(atok) > 3 and atok not in STOPWORDS:
                    c["addr_num_idx"][(atok, num)].append(idx)

# Test function
def evaluate_matching_pipeline(v11_strict=True):
    MAX_NAME_POSTINGS = 200
    MAX_ADDR_POSTINGS = 50
    MAX_MATCHES = 4

    predictions = {}
    
    for qid in sample_qids:
        q_name_raw, q_addr_raw, ctry = s1_records[qid]
        c_data = countries.get(ctry)
        if not c_data:
            predictions[qid] = []
            continue

        q_name = clean_text(q_name_raw)
        q_addr = clean_text(q_addr_raw)
        q_comp = compress_name(q_name_raw)
        q_tokens = get_tokens(q_name)
        q_nums = get_nums(q_addr)

        comp_idx = c_data["comp_idx"]
        name_idx = c_data["name_idx"]
        addr_num_idx = c_data["addr_num_idx"]

        cands = set()
        if q_comp and q_comp in comp_idx:
            cands.update(comp_idx[q_comp])
        if q_tokens:
            sorted_toks = sorted(q_tokens, key=lambda t: len(name_idx.get(t, ())))
            for t in sorted_toks[:2]:
                postings = name_idx.get(t)
                if postings and len(postings) <= MAX_NAME_POSTINGS:
                    cands.update(postings)
        if q_nums:
            for num in list(q_nums)[:2]:
                for atok in q_addr.split()[:3]:
                    if len(atok) > 3 and atok not in STOPWORDS:
                        postings = addr_num_idx.get((atok, num))
                        if postings and len(postings) <= MAX_ADDR_POSTINGS:
                            cands.update(postings)

        if not cands:
            predictions[qid] = []
            continue

        t_names = c_data["name"]
        t_addrs = c_data["addr"]
        t_comps = c_data["comp"]
        t_nums = c_data["nums"]
        t_ids = c_data["ids"]

        matches = []
        for cid in cands:
            t_comp = t_comps[cid]
            t_n = t_names[cid]
            t_a = t_addrs[cid]
            t_num = t_nums[cid]

            # Hard building number conflict rejection
            if q_nums and t_num and not (q_nums & t_num):
                continue

            if q_comp and t_comp and q_comp == t_comp:
                matches.append((1.0, t_ids[cid]))
                continue

            nsim = fuzz.token_sort_ratio(q_name, t_n)
            asim = fuzz.token_sort_ratio(q_addr, t_a)
            num_match = bool(q_nums & t_num) if (q_nums and t_num) else False
            score = nsim * 0.6 + asim * 0.4

            is_match = False
            if nsim >= 70 and (asim >= 40 or num_match or not t_a):
                is_match = True
            elif nsim >= 55 and asim >= 75 and num_match:
                is_match = True
            elif nsim >= 85:
                is_match = True

            if is_match:
                matches.append((score, t_ids[cid]))

        matches.sort(key=lambda x: x[0], reverse=True)
        predictions[qid] = [eid for sc, eid in matches[:MAX_MATCHES]]

    # Compute F0.5
    beta = 0.5
    b2 = beta ** 2
    scores = []
    tp_tot = fp_tot = fn_tot = 0
    singletons_correct = 0
    singletons_ruined = 0

    for qid in sample_qids:
        preds = set(predictions[qid])
        gts = set(gt_dict[qid])
        tp = len(preds & gts)
        fp = len(preds - gts)
        fn = len(gts - preds)
        tp_tot += tp
        fp_tot += fp
        fn_tot += fn
        
        if len(gts) == 0:
            if len(preds) == 0:
                singletons_correct += 1
                scores.append(1.0)
            else:
                singletons_ruined += 1
                scores.append(0.0)
        else:
            denom = (1 + b2) * tp + b2 * fn + fp
            scores.append((1 + b2) * tp / denom if denom > 0 else 0.0)

    macro_f05 = np.mean(scores)
    prec = tp_tot / max(1, tp_tot + fp_tot)
    rec = tp_tot / max(1, tp_tot + fn_tot)
    print("=" * 60)
    print(f"V11 EVALUATION RESULT:")
    print(f"Macro F_0.5 Score:     {macro_f05:.4f}")
    print(f"Micro Precision:       {prec:.4f} (TP: {tp_tot}, FP: {fp_tot})")
    print(f"Micro Recall:          {rec:.4f} (TP: {tp_tot}, FN: {fn_tot})")
    print(f"Singletons:            {singletons_correct}/{s_count} correct ({(singletons_correct/s_count)*100:.1f}%), {singletons_ruined} false alarms")
    print("=" * 60)
    return predictions

preds = evaluate_matching_pipeline()
