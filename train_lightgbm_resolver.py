import sys, re, unicodedata, time
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import lightgbm as lgb
from anyascii import anyascii
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

TRAIN_DIR = Path("student_resource/dataset/train")
MODEL_DIR = Path("student_resource/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS_REGEX = re.compile(r'\.(com|net|org|co\.in|in|fr|co|io|info|biz)\b', re.IGNORECASE)
LEGAL_SUFFIXES = {
    "corp", "corporation", "corporate", "pvt", "private", "ltd", "limited",
    "llc", "inc", "incorporated", "co", "company", "sarl", "sas", "llp", "sa", "gmbh",
    "sasu", "eurl", "dba", "services", "holdings", "group", "groupe", "center", "enterprises"
}
STOPWORDS = {"the", "and", "of", "a", "an", "in", "to", "for", "at", "on", "by", "null"}

def normalize_text(text: str) -> str:
    if not text:
        return ""
    # 1. Transliterate all scripts (Devanagari, Bengali, Tamil, etc.) to ASCII
    text = anyascii(str(text)).lower()
    # 2. Strip leading zeros from numeric sequences: '0337' -> '337'
    text = re.sub(r'\b0+(?=\d)', '', text)
    # 3. Standardize address hashes and unit symbols: '##20' -> '20'
    text = re.sub(r'[#,\-\/]', ' ', text)
    # 4. Remove standard domains
    text = DOMAINS_REGEX.sub('', text)
    # 5. Collapse spaces
    return re.sub(r'\s+', ' ', text).strip()

def compress_name(n: str) -> str:
    n = normalize_text(n)
    n = re.sub(r'[^a-z0-9]', '', n)
    for suf in (
        "corporation", "corporate", "private", "limited", "holdings", "services", 
        "enterprises", "company", "corp", "pvt", "ltd", "llc", "inc", "gmbh", 
        "sarl", "sas", "sasu", "eurl", "dba", "group", "groupe", "center"
    ):
        if n.endswith(suf):
            n = n[:-len(suf)]
    return n

def get_tokens(s: str):
    return frozenset(t for t in s.split() if t not in LEGAL_SUFFIXES and t not in STOPWORDS and len(t) > 1)

def get_nums(s: str):
    raw = re.findall(r'\b\d+\b', s)
    return frozenset(n.lstrip("0") or "0" for n in raw if len(n) >= 2)

def compute_entity_f_beta(pred_set, true_set, beta=0.5):
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    if len(pred_set) == 0 or tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta_sq = beta ** 2
    return ((1 + beta_sq) * precision * recall) / ((beta_sq * precision) + recall)

def extract_features(s1_name, s1_addr, s1_comp, s1_nums, s1_tokens,
                     tgt_name, tgt_addr, tgt_comp, tgt_nums, tgt_tokens):
    # RapidFuzz SIMD features
    jw = float(JaroWinkler.similarity(s1_name, tgt_name))
    tsort_name = fuzz.token_sort_ratio(s1_name, tgt_name) / 100.0
    tset_name = fuzz.token_set_ratio(s1_name, tgt_name) / 100.0
    tsort_addr = fuzz.token_sort_ratio(s1_addr, tgt_addr) / 100.0
    
    # Exact / compressed name match
    comp_match = 1.0 if (s1_comp and tgt_comp and s1_comp == tgt_comp) else 0.0
    
    # Numeric address agreement
    num_inter = len(s1_nums & tgt_nums) if s1_nums and tgt_nums else 0
    num_match = 1.0 if num_inter > 0 else 0.0
    
    # Token Jaccard
    n_inter = len(s1_tokens & tgt_tokens)
    n_union = len(s1_tokens | tgt_tokens)
    name_jacc = n_inter / n_union if n_union > 0 else 0.0
    
    len_diff = abs(len(s1_name) - len(tgt_name))
    
    return [jw, tsort_name, tset_name, tsort_addr, comp_match, num_match, name_jacc, len_diff]

FEATURE_NAMES = [
    "jw", "tsort_name", "tset_name", "tsort_addr",
    "comp_match", "num_match", "name_jacc", "len_diff"
]

def main():
    print("=" * 65)
    print("Training High-Precision GBDT Resolver (Stage 2 & 3)")
    print("=" * 65)
    
    # 1. Load Ground Truth for 10,000 queries (Train: 8,000, Val: 2,000)
    print("Loading 10,000 ground truth queries...")
    all_gt = {}
    with open(TRAIN_DIR / "train_ground_truth.tsv", "r", encoding="utf-8") as f:
        next(f)
        for i, line in enumerate(f):
            if i >= 10000: break
            p = line.strip().split("\t")
            all_gt[p[0]] = set(p[1].split(",")) if len(p) > 1 and p[1] else set()
            
    # Load Source 1 entities
    s1_entities = {}
    needed_s1 = set(all_gt.keys())
    with open(TRAIN_DIR / "train_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.strip().split("\t")
            if p[0] in needed_s1:
                norm_name = normalize_text(p[1])
                norm_addr = normalize_text(p[2]) if len(p) > 2 else ""
                ctry = p[3].strip() if len(p) > 3 else "Unknown"
                s1_entities[p[0]] = {
                    "name": norm_name,
                    "addr": norm_addr,
                    "comp": compress_name(p[1]),
                    "nums": get_nums(norm_addr),
                    "tokens": get_tokens(norm_name),
                    "ctry": ctry
                }
                if len(s1_entities) >= len(needed_s1): break

    needed_targets = {eid for matches in all_gt.values() for eid in matches}
    print(f"Loaded {len(s1_entities):,} S1 entities. Found {len(needed_targets):,} true target IDs.")
    
    # Load targets + 250,000 negative distractors
    target_entities = {}
    loaded_distractors = 0
    distractor_cap = 250000
    
    for src in ["train_source2.tsv", "train_source3.tsv"]:
        print(f"Loading from {src}...")
        with open(TRAIN_DIR / src, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.strip().split("\t")
                eid = p[0]
                if eid in needed_targets or loaded_distractors < distractor_cap:
                    norm_name = normalize_text(p[1])
                    norm_addr = normalize_text(p[2]) if len(p) > 2 else ""
                    ctry = p[3].strip() if len(p) > 3 else "Unknown"
                    target_entities[eid] = {
                        "name": norm_name,
                        "addr": norm_addr,
                        "comp": compress_name(p[1]),
                        "nums": get_nums(norm_addr),
                        "tokens": get_tokens(norm_name),
                        "ctry": ctry
                    }
                    if eid not in needed_targets:
                        loaded_distractors += 1

    print(f"Total target space indexed: {len(target_entities):,} records (including {loaded_distractors:,} negative distractors)!")

    # Build Blocking Inverted Indexes
    print("Building blocking indexes...")
    comp_idx = defaultdict(list)
    name_idx = defaultdict(list)
    addr_num_idx = defaultdict(list)
    
    for tid, tgt in target_entities.items():
        c = tgt["ctry"]
        if tgt["comp"]:
            comp_idx[(c, tgt["comp"])].append(tid)
        for t in tgt["tokens"]:
            name_idx[(c, t)].append(tid)
        for num in tgt["nums"]:
            for atok in tgt["addr"].split():
                if len(atok) > 3 and atok not in {"road", "street", "avenue", "city", "state", "near"}:
                    addr_num_idx[(c, atok, num)].append(tid)

    # Train / Val Split
    query_ids = list(s1_entities.keys())
    train_ids = query_ids[:8000]
    val_ids = query_ids[8000:]
    
    print(f"\nGenerating training pairs with Hard Negative Mining (8,000 train queries)...")
    X_train = []
    y_train = []
    
    for s1_id in train_ids:
        s1 = s1_entities[s1_id]
        true_m = all_gt[s1_id]
        ctry = s1["ctry"]
        
        # Blocker
        cands = set()
        if s1["comp"]:
            cands.update(comp_idx.get((ctry, s1["comp"]), ()))
        for t in sorted(s1["tokens"], key=lambda x: len(name_idx.get((ctry, x), ())))[:2]:
            postings = name_idx.get((ctry, t), ())
            if len(postings) <= 150:
                cands.update(postings)
        for num in s1["nums"]:
            for atok in s1["addr"].split():
                if len(atok) > 3 and atok not in {"road", "street", "avenue", "city", "state", "near"}:
                    postings = addr_num_idx.get((ctry, atok, num), ())
                    if len(postings) <= 30:
                        cands.update(postings)
                        
        # Ensure true matches in target_entities are included
        for tid in true_m:
            if tid in target_entities:
                cands.add(tid)
                
        for cid in cands:
            tgt = target_entities[cid]
            feat = extract_features(
                s1["name"], s1["addr"], s1["comp"], s1["nums"], s1["tokens"],
                tgt["name"], tgt["addr"], tgt["comp"], tgt["nums"], tgt["tokens"]
            )
            label = 1 if cid in true_m else 0
            X_train.append(feat)
            y_train.append(label)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"Generated {len(y_train):,} training pairs ({y_train.sum():,} positive, {len(y_train)-y_train.sum():,} hard negatives)!")

    # Train LightGBM
    print("\nTraining LightGBM model...")
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES, free_raw_data=False)
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_bin": 255,
        "is_unbalance": True,
        "n_jobs": -1,
        "verbose": -1
    }
    model = lgb.train(params, dtrain, num_boost_round=250)
    
    # Save model
    model_path = MODEL_DIR / "lgb_entity_resolver.txt"
    model.save_model(str(model_path))
    print(f"Model saved to: {model_path}")
    
    # Feature Importance
    print("\nFeature Importances:")
    imp = model.feature_importance(importance_type="gain")
    for name, val in sorted(zip(FEATURE_NAMES, imp), key=lambda x: x[1], reverse=True):
        print(f"  {name:15s}: {val:.1f}")

    # 4. Calibrate Threshold on 2,000 Validation Queries
    print(f"\nCalibrating optimal decision threshold on 2,000 validation queries...")
    val_cand_cache = []
    
    for s1_id in val_ids:
        s1 = s1_entities[s1_id]
        true_m = all_gt[s1_id]
        ctry = s1["ctry"]
        
        cands = set()
        if s1["comp"]:
            cands.update(comp_idx.get((ctry, s1["comp"]), ()))
        for t in sorted(s1["tokens"], key=lambda x: len(name_idx.get((ctry, x), ())))[:2]:
            postings = name_idx.get((ctry, t), ())
            if len(postings) <= 150:
                cands.update(postings)
        for num in s1["nums"]:
            for atok in s1["addr"].split():
                if len(atok) > 3 and atok not in {"road", "street", "avenue", "city", "state", "near"}:
                    postings = addr_num_idx.get((ctry, atok, num), ())
                    if len(postings) <= 30:
                        cands.update(postings)
                        
        cand_list = list(cands)
        if not cand_list:
            val_cand_cache.append((s1_id, true_m, [], []))
            continue
            
        feats = []
        for cid in cand_list:
            tgt = target_entities[cid]
            feat = extract_features(
                s1["name"], s1["addr"], s1["comp"], s1["nums"], s1["tokens"],
                tgt["name"], tgt["addr"], tgt["comp"], tgt["nums"], tgt["tokens"]
            )
            feats.append(feat)
            
        probs = model.predict(np.array(feats, dtype=np.float32))
        val_cand_cache.append((s1_id, true_m, cand_list, probs))

    best_threshold = 0.50
    best_f05 = 0.0
    
    print("\nThreshold Grid Search for Macro F_0.5:")
    for th in np.arange(0.50, 0.95, 0.05):
        scores = []
        tot_tp, tot_fp, tot_fn = 0, 0, 0
        
        for s1_id, true_m, cand_list, probs in val_cand_cache:
            preds = {cid for cid, p in zip(cand_list, probs) if p >= th}
            f05 = compute_entity_f_beta(preds, true_m)
            scores.append(f05)
            
            tp = len(preds & true_m)
            fp = len(preds - true_m)
            fn = len(true_m - preds)
            tot_tp += tp
            tot_fp += fp
            tot_fn += fn
            
        macro = sum(scores) / len(scores)
        prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0
        rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0
        print(f"  Threshold {th:.2f} -> Macro F_0.5 = {macro:.4f} | Precision = {prec:.4f} | Recall = {rec:.4f}")
        
        if macro > best_f05:
            best_f05 = macro
            best_threshold = th
            
    print("=" * 65)
    print(f"OPTIMAL THRESHOLD LOCKED: {best_threshold:.2f} WITH MACRO F_0.5 = {best_f05:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    main()
