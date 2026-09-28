import sys, re
import pandas as pd
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from anyascii import anyascii
from rapidfuzz import fuzz

DATA_DIR = Path('student_resource/dataset/train')
gt = pd.read_csv(DATA_DIR / 'train_ground_truth.tsv', sep='\t', nrows=50)
gt_dict = {}
all_tgts = set()
for _, r in gt.iterrows():
    qid = r['source1_entity_id']
    if pd.isna(r['matched_entity_ids']):
        gt_dict[qid] = []
    else:
        tgts = [x.strip() for x in str(r['matched_entity_ids']).split(',') if x.strip()]
        gt_dict[qid] = tgts
        all_tgts.update(tgts)

# Load S1
s1_map = {}
with open(DATA_DIR / 'train_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in gt_dict:
            s1_map[p[0]] = (p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else '')

# Load S2 and S3 targets
target_records = {}
for fn in ['train_source2.tsv', 'train_source3.tsv']:
    with open(DATA_DIR / fn, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if p[0] in all_tgts:
                target_records[p[0]] = (p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else '')

print(f"Loaded {len(s1_map)} queries and {len(target_records)} targets.")

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
    return frozenset(n.lstrip("0") or "0" for n in raw if len(n) >= 1)

def match_pair(qname, qaddr, tname, taddr):
    # Transliterated / cleaned
    qn_clean = clean_text(qname)
    tn_clean = clean_text(tname)
    qa_clean = clean_text(qaddr)
    ta_clean = clean_text(taddr)

    qcomp = compress_name(qname)
    tcomp = compress_name(tname)

    qnums = get_nums(qaddr)
    tnums = get_nums(taddr)

    # Number overlap
    num_agree = bool(qnums and tnums and (qnums & tnums))
    num_conflict = bool(qnums and tnums and not (qnums & tnums))

    # Name similarity
    nsim_sort = fuzz.token_sort_ratio(qn_clean, tn_clean)
    nsim_set = fuzz.token_set_ratio(qn_clean, tn_clean)
    nsim = max(nsim_sort, nsim_set)

    # Address similarity (both sort and set)
    asim_sort = fuzz.token_sort_ratio(qa_clean, ta_clean) if (qa_clean and ta_clean) else 0
    asim_set = fuzz.token_set_ratio(qa_clean, ta_clean) if (qa_clean and ta_clean) else 0
    asim = max(asim_sort, asim_set)

    # Address text similarity without numbers (street/city match)
    qa_nonum = re.sub(r'\d+', ' ', qa_clean).strip()
    ta_nonum = re.sub(r'\d+', ' ', ta_clean).strip()
    street_sim = fuzz.token_set_ratio(qa_nonum, ta_nonum) if (qa_nonum and ta_nonum) else 0

    # Number typo check: e.g. 8706 vs 870, 769 vs 69, 9236 vs 9238
    # If street matches strongly (>=80) and name matches strongly (>=80), number diff is just a typo!
    if num_conflict:
        is_num_typo = False
        if nsim >= 80 and street_sim >= 75:
            # Check if any pair of numbers has Levenshtein distance <= 1 or substring
            for qn in qnums:
                for tn in tnums:
                    if len(qn) >= 2 and len(tn) >= 2:
                        if qn in tn or tn in qn or abs(len(qn) - len(tn)) <= 1:
                            if fuzz.ratio(qn, tn) >= 66:
                                is_num_typo = True
                                break
                if is_num_typo: break
        if not is_num_typo:
            return False, 0.0, "num_conflict"

    # 1. Exact compressed name match (handles domains, punctuation, legal suffixes)
    if qcomp and tcomp and qcomp == tcomp:
        return True, 100.0, "exact_comp"

    # 2. Corrupted / masked / Indic name, but address matches with building number
    if num_agree and asim >= 50:
        return True, 90.0, "addr_agree_num_match"

    # 3. High address similarity (even without numbers if address is long)
    if asim >= 75 and len(qa_clean) > 15:
        return True, 85.0, "high_addr_match"

    # 4. High name similarity with corroboration OR empty target address
    if nsim >= 82:
        if not ta_clean or not qa_clean or asim >= 25 or num_agree or street_sim >= 50:
            return True, 80.0, "high_name_match"

    # 5. Moderate name + Moderate address
    if nsim >= 60 and (asim >= 40 or num_agree or street_sim >= 60):
        return True, 75.0, "joint_name_addr"

    # 6. Suffix / variation when target address is empty
    if not ta_clean and nsim >= 75:
        return True, 70.0, "name_empty_addr"

    return False, 0.0, "no_match"

tp = fp = fn = 0
for qid in gt_dict:
    qname, qaddr, qctry = s1_map.get(qid, ('', '', ''))
    true_tgts = set(gt_dict[qid])
    for tid, (tname, taddr, tctry) in target_records.items():
        if tid in true_tgts: # only evaluate against candidates for this entity
            is_m, score, reason = match_pair(qname, qaddr, tname, taddr)
            if is_m:
                tp += 1
            else:
                fn += 1
                print(f"FN on {qid} -> {tid} ({reason}):")
                print(f"  Q: '{qname}' | '{qaddr}'")
                print(f"  T: '{tname}' | '{taddr}'")

print(f"\nRecall on true matches: {tp}/{tp+fn} ({tp/(tp+fn)*100:.2f}%)")
