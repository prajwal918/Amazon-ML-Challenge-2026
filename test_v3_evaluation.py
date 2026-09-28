import sys, re, unicodedata, time
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(r".")
TRAIN_DIR = BASE_DIR / "student_resource" / "dataset" / "train"

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

DOMAINS = (".com", ".net", ".org", ".co.in", ".in", ".fr", ".co", ".io", ".info", ".biz")
LEGAL_SUFFIXES = (
    "corporation", "corporate", "private", "limited", "holdings", "services", 
    "enterprises", "company", "corp", "pvt", "ltd", "llc", "inc", "gmbh", 
    "sarl", "sas", "sasu", "eurl", "dba", "group", "groupe", "center"
)

def compress_name(n: str) -> str:
    n = strip_accents(str(n).lower().strip())
    for ext in DOMAINS:
        if n.endswith(ext):
            n = n[:-len(ext)]
    n = re.sub(r"[^a-z0-9]", "", n)
    for suf in LEGAL_SUFFIXES:
        if n.endswith(suf):
            n = n[:-len(suf)]
    return n

STOPWORDS = {"the", "and", "of", "a", "an", "in", "to", "for", "at", "on", "by", "null"}
ADDRESS_ABBREV = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "apt": "apartment", "hwy": "highway",
}

def clean_name(name: str):
    name = strip_accents(str(name).lower())
    name = name.replace("&", " and ").replace("+", " and ")
    name = re.sub(r"[^\w\s]", " ", name)
    return frozenset(t for t in name.split() if t not in LEGAL_SUFFIXES and t not in STOPWORDS and len(t) > 1)

def clean_addr(addr: str):
    addr = strip_accents(str(addr).lower())
    addr = re.sub(r"[^\w\s]", " ", addr)
    toks = [ADDRESS_ABBREV.get(t, t) for t in addr.split()]
    return frozenset(t for t in toks if t not in STOPWORDS and len(t) > 1)

def clean_nums(addr: str):
    raw = re.findall(r"\b\d+\b", str(addr))
    return frozenset(n for n in raw if len(n) >= 2)

def is_latin(text: str) -> bool:
    return all(ord(c) < 256 for c in text if c.isalpha())

# Load 1000 GT queries
print("Loading 1,000 training ground truth queries...")
gt = {}
with open(TRAIN_DIR / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        if i >= 1000: break
        p = line.strip().split("\t")
        gt[p[0]] = set(p[1].split(",")) if len(p) > 1 and p[1] else set()

needed_s1 = set(gt.keys())
s1_entities = {}
with open(TRAIN_DIR / "train_source1.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.strip().split("\t")
        if p[0] in needed_s1:
            s1_entities[p[0]] = {
                "name": p[1],
                "addr": p[2] if len(p) > 2 else "",
                "ctry": p[3].strip() if len(p) > 3 else "Unknown"
            }
            if len(s1_entities) >= len(needed_s1):
                break

needed_targets = {eid for matches in gt.values() for eid in matches}
print(f"Loaded {len(s1_entities)} S1 entities. Need {len(needed_targets)} matching targets.")

# Collect targets from S2 and S3
target_entities = {}
for src in ["train_source2.tsv", "train_source3.tsv"]:
    with open(TRAIN_DIR / src, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.strip().split("\t")
            if p[0] in needed_targets:
                target_entities[p[0]] = {
                    "id": p[0],
                    "name": p[1],
                    "addr": p[2] if len(p) > 2 else "",
                    "ctry": p[3].strip() if len(p) > 3 else "Unknown"
                }

print(f"Collected {len(target_entities)} true target entities.")
print("Testing V3 rule matching on these ground truth pairs...")

tp_count = 0
fn_count = 0
for s1_id, s1 in s1_entities.items():
    true_matches = gt[s1_id]
    if not true_matches:
        continue
    
    s1_comp = compress_name(s1["name"])
    s1_n = clean_name(s1["name"])
    s1_a = clean_addr(s1["addr"])
    s1_nums = clean_nums(s1["addr"])
    s1_lat = is_latin(s1["name"])
    
    for tid in true_matches:
        if tid not in target_entities:
            continue
        tgt = target_entities[tid]
        t_comp = compress_name(tgt["name"])
        t_n = clean_name(tgt["name"])
        t_a = clean_addr(tgt["addr"])
        t_nums = clean_nums(tgt["addr"])
        t_lat = is_latin(tgt["name"])
        
        # Check rule match
        is_match = False
        reason = ""
        
        # Rule 1: Exact compressed name match
        if s1_comp and t_comp and s1_comp == t_comp:
            is_match = True
            reason = "Exact compressed name"
        
        # Rule 2: Standard Name Jaccard + Address
        if not is_match:
            n_inter = len(s1_n & t_n)
            n_union = len(s1_n | t_n)
            n_j = n_inter / n_union if n_union > 0 else 0.0
            
            a_inter = len(s1_a & t_a)
            a_union = len(s1_a | t_a)
            a_j = a_inter / a_union if a_union > 0 else 0.0
            has_num = bool(s1_nums & t_nums)
            
            if n_j >= 0.50 and (a_j >= 0.12 or has_num or not bool(t_a)):
                is_match = True
                reason = f"Name Jaccard {n_j:.2f} + Addr {a_j:.2f}"
            elif n_j >= 0.30 and (a_j >= 0.25 or has_num):
                is_match = True
                reason = f"Moderate Name {n_j:.2f} + Strong Addr {a_j:.2f}"
            elif not t_lat and s1["ctry"] == "India" and (a_j >= 0.40 or (has_num and a_j >= 0.25)):
                is_match = True
                reason = f"Cross-script Indian match Addr {a_j:.2f}"
            elif has_num and a_j >= 0.55:
                is_match = True
                reason = f"Exact building number + street match {a_j:.2f}"
        
        if is_match:
            tp_count += 1
        else:
            fn_count += 1

total_tested = tp_count + fn_count
print(f"Results on true pairs: {tp_count}/{total_tested} matches caught ({tp_count/max(1, total_tested)*100:.2f}% RECALL!)")
