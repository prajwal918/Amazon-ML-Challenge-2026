#!/usr/bin/env python3
"""
Quick threshold swap: reads raw scores and generates a new submission file
at a specified threshold — instant, no inference needed.
Usage:
  python quick_threshold.py 0.75 0.65           # base_th sib_th
  python quick_threshold.py 0.80                # base_th (sib_th = base_th - 0.05)
  python quick_threshold.py 0.80 0.60 --version v22  # from specific version's scores
"""
import sys, os, time, shutil, zipfile
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_threshold.py <base_th> [sib_th] [--version v20|v21|v22]")
        return

    base_th = float(sys.argv[1])
    sib_th = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith('-') else max(0.25, base_th - 0.05)
    version = "v22"
    if "--version" in sys.argv:
        idx = sys.argv.index("--version")
        if idx + 1 < len(sys.argv): version = sys.argv[idx+1]

    raw_file = Path(f"output/raw_scores_{version}.tsv")
    if not raw_file.exists():
        # try others
        for v in ["v22","v21","v20"]:
            f = Path(f"output/raw_scores_{v}.tsv")
            if f.exists():
                raw_file = f; version = v; break
    if not raw_file.exists():
        print(f"ERROR: No raw scores found. Run inference first."); return

    print(f"Loading raw scores from {raw_file}...")
    t0 = time.time()
    s1_order = []
    score_dict = {}
    with open(raw_file, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            qid = p[0]; s1_order.append(qid)
            if len(p) > 1 and p[1]:
                pairs = []
                for entry in p[1].split(";"):
                    if ":" in entry:
                        tid, prob = entry.rsplit(":", 1)
                        pairs.append((tid, float(prob)))
                score_dict[qid] = sorted(pairs, key=lambda x: -x[1])
            else:
                score_dict[qid] = []

    print(f"  Loaded {len(s1_order):,} queries in {time.time()-t0:.1f}s")
    print(f"  Applying BaseTh={base_th:.2f}, SibTh={sib_th:.2f}...")

    out_path = Path("output/matching_results.tsv")
    n_matched = 0
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for qid in s1_order:
            pairs = score_dict[qid]
            chosen = []
            if pairs and pairs[0][1] >= base_th:
                chosen.append(pairs[0][0])
                for tid, prob in pairs[1:]:
                    if prob >= sib_th: chosen.append(tid)
                    else: break
            if chosen: n_matched += 1
            f.write(f"{qid}\t{','.join(chosen)}\n")

    pct = n_matched / max(1, len(s1_order)) * 100
    ns = len(s1_order) - n_matched
    print(f"  Result: {n_matched:,} matched ({pct:.1f}%) | {ns:,} singletons ({100-pct:.1f}%)")
    print(f"  Written → {out_path}")

    # Also package zip
    print("\nPackaging ZIP...")
    BASE_DIR = Path(r".")
    DESKTOP = Path(r"~/Desktop")
    zip_path = BASE_DIR / "Brad_Pitt_submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.write(BASE_DIR / "output" / "matching_results.tsv", "output/matching_results.tsv")
        zf.write(BASE_DIR / "output" / "candidate_pairs.tsv", "output/candidate_pairs.tsv")
        if (BASE_DIR / "Documentation_template.md").exists():
            zf.write(BASE_DIR / "Documentation_template.md", "Documentation_template.md")
        if (BASE_DIR / "documentation.md").exists():
            zf.write(BASE_DIR / "documentation.md", "documentation.md")
        code_dir = BASE_DIR / "code" / "business_entity_resolution"
        if code_dir.exists():
            for p in code_dir.rglob("*"):
                if p.is_file() and "__pycache__" not in str(p):
                    zf.write(p, str(p.relative_to(BASE_DIR)).replace("\\","/"))
    shutil.copyfile(zip_path, DESKTOP / "Brad_Pitt_submission.zip")
    mb = zip_path.stat().st_size / 1024 / 1024
    print(f"  Brad_Pitt_submission.zip ({mb:.1f} MB) → Desktop")
    print(f"  Total: {time.time()-t0:.1f}s — READY TO SUBMIT!")

if __name__ == "__main__":
    main()
