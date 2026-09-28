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

# Load 1000 GT
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

# Load the needed targets
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

# Build mini-index of these targets
comp_idx = defaultdict(list)
name_tok_idx = defaultdict(list)
addr_num_idx = defaultdict(list)

for tid, tgt in target_entities.items():
    cn = compress_name(tgt["name"])
    if cn:
        comp_idx[cn].append(tid)
    
    for t in clean_name(tgt["name"]):
        name_tok_idx[t].append(tid)
        
    nums = clean_nums(tgt["addr"])
    atoks = clean_addr(tgt["addr"])
    for num in nums:
        for atok in atoks:
            addr_num_idx[(tgt["ctry"], atok, num)].append(tid)

# Now test blocking candidate recall for S1 queries
retrieved_true = 0
total_true = sum(len(matches) for matches in gt.values())

for s1_id, s1 in s1_entities.items():
    true_m = gt[s1_id]
    if not true_m:
        continue
        
    s1_comp = compress_name(s1["name"])
    s1_n = clean_name(s1["name"])
    s1_a = clean_addr(s1["addr"])
    s1_nums = clean_nums(s1["addr"])
    ctry = s1["ctry"]
    
    candidates = set()
    # 1. Comp name
    if s1_comp and s1_comp in comp_idx:
        candidates.update(comp_idx[s1_comp])
        
    # 2. Name tokens
    for t in s1_n:
        if t in name_tok_idx:
            candidates.update(name_tok_idx[t])
            
    # 3. Addr num + token
    for num in s1_nums:
        for atok in s1_a:
            key = (ctry, atok, num)
            if key in addr_num_idx:
                candidates.update(addr_num_idx[key])
                
    hits = len(candidates & true_m)
    retrieved_true += hits

print(f"Total True Targets: {total_true}")
print(f"Candidates Retrieved: {retrieved_true}/{total_true} ({retrieved_true/total_true*100:.2f}% BLOCKING RECALL!)")
