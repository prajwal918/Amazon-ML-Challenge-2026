import sys, io
import pandas as pd
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path('student_resource/dataset/train')
gt = pd.read_csv(DATA_DIR / 'train_ground_truth.tsv', sep='\t', nrows=25)
gt_qids = set(gt['source1_entity_id'])

# Read matching S1 records
s1_map = {}
with open(DATA_DIR / 'train_source1.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\r\n').split('\t')
        if p[0] in gt_qids:
            s1_map[p[0]] = (p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else '')
            if len(s1_map) == len(gt_qids):
                break

# Targets to find
all_tgts = set()
for _, r in gt.iterrows():
    if pd.notna(r['matched_entity_ids']):
        all_tgts.update([x.strip() for x in str(r['matched_entity_ids']).split(',') if x.strip()])

s23_map = {}
for fn in ['train_source2.tsv', 'train_source3.tsv']:
    with open(DATA_DIR / fn, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\r\n').split('\t')
            if p[0] in all_tgts:
                s23_map[p[0]] = (p[1] if len(p)>1 else '', p[2] if len(p)>2 else '', p[3] if len(p)>3 else '')
                if len(s23_map) == len(all_tgts):
                    break

print("=== REAL GROUND TRUTH SAMPLES ===")
for idx, row in gt.head(8).iterrows():
    qid = row['source1_entity_id']
    m = str(row['matched_entity_ids'])
    targets = [x.strip() for x in m.split(',') if x.strip()] if pd.notna(row['matched_entity_ids']) else []
    qname, qaddr, qctry = s1_map.get(qid, ('', '', ''))
    print(f"\n--- S1 [{qid}] ({qctry}):")
    print(f"    Name:    '{qname}'")
    print(f"    Address: '{qaddr}'")
    print(f"    Matched Targets ({len(targets)}):")
    for tid in targets:
        tname, taddr, tctry = s23_map.get(tid, ('MISSING', '', ''))
        print(f"      -> [{tid}]: '{tname}' | '{taddr}'")
