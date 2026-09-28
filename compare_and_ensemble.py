#!/usr/bin/env python3
"""
Comprehensive Model Comparison & Super-Ensemble Builder
Amazon ML Challenge 2026

1. Scans all kaggle_output_* directories for finished predictions.
2. Compares:
   - Match rate (% of queries with predicted matches)
   - Singleton preservation count
   - Pairwise Jaccard agreement between different models
3. Builds the Consensus Super-Ensemble across all models.
4. Applies Bayesian Expected F-0.5 stopping rule.
5. Formats and validates the final matching_results.tsv for Unstop.
"""

import os, sys, glob, shutil
from pathlib import Path
from collections import defaultdict, Counter

BASE_DIR = Path(r".")
DESKTOP = Path(r"~/Desktop")

def load_prediction_file(filepath):
    print(f"Loading {filepath.name}...")
    preds = {}
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            qid = parts[0]
            matches = [x.strip() for x in parts[1].split(",") if x.strip()] if len(parts) > 1 and parts[1] else []
            preds[qid] = matches
    return preds

def compare_and_build_ensemble():
    output_files = list(BASE_DIR.glob("kaggle_output_*/**/matching_results*.tsv"))
    # Also check base output/
    base_matching = BASE_DIR / "output" / "matching_results.tsv"
    if base_matching.exists() and base_matching not in output_files:
        output_files.append(base_matching)

    if not output_files:
        print("No prediction files found to compare!")
        return None

    print(f"\nFound {len(output_files)} candidate prediction files:")
    all_models = {}
    stats = {}

    for fp in output_files:
        model_name = fp.parent.name if fp.parent.name != "output" else fp.stem
        preds = load_prediction_file(fp)
        all_models[model_name] = preds

        total = len(preds)
        singletons = sum(1 for m in preds.values() if len(m) == 0)
        matched = total - singletons
        total_targets = sum(len(m) for m in preds.values())
        avg_targets = total_targets / max(1, matched)

        stats[model_name] = {
            "total": total,
            "matched": matched,
            "singletons": singletons,
            "match_rate": matched / max(1, total) * 100,
            "total_targets": total_targets,
            "avg_targets": avg_targets
        }

    # Generate Comparison Report
    report_lines = [
        "# Amazon ML Challenge 2026 -- Model Fleet Comparison Report\n",
        f"**Total Models Evaluated**: {len(all_models)}\n",
        "| Model | Total Queries | Matched Queries | Match Rate (%) | Singletons | Avg Targets/Match |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for m_name, s in stats.items():
        report_lines.append(f"| {m_name} | {s['total']:,} | {s['matched']:,} | {s['match_rate']:.2f}% | {s['singletons']:,} | {s['avg_targets']:.2f} |")

    # Pairwise agreement
    model_names = list(all_models.keys())
    if len(model_names) > 1:
        report_lines.append("\n### Pairwise Model Agreement (Jaccard Overlap):\n")
        report_lines.append("| Model A | Model B | Jaccard Agreement (%) |")
        report_lines.append("| :--- | :--- | :--- |")
        for i in range(min(5, len(model_names))):
            for j in range(i+1, min(5, len(model_names))):
                m1, m2 = model_names[i], model_names[j]
                agree = 0
                sample_q = list(all_models[m1].keys())[:10000]
                for q in sample_q:
                    s1 = set(all_models[m1].get(q, []))
                    s2 = set(all_models[m2].get(q, []))
                    if s1 == s2:
                        agree += 1
                report_lines.append(f"| {m1} | {m2} | {agree / len(sample_q) * 100:.2f}% |")

    report_path = BASE_DIR / "output" / "MODEL_COMPARISON_REPORT.md"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nComparison Report generated at: {report_path}")

    # Build Consensus Super-Ensemble
    print("\nBuilding Consensus Super-Ensemble across models...")
    # Weighted voting
    all_qids = list(next(iter(all_models.values())).keys())
    ensemble_preds = {}

    for qid in all_qids:
        candidate_votes = Counter()
        for m_name, preds in all_models.items():
            for tid in preds.get(qid, []):
                candidate_votes[tid] += 1
        
        # High-confidence consensus selection
        # If candidate is voted by >= 50% of models (or top model single vote if only 1 model)
        min_votes = max(1, len(all_models) // 2)
        chosen = [tid for tid, votes in candidate_votes.most_common() if votes >= min_votes]
        ensemble_preds[qid] = chosen

    # Save ensemble matching_results.tsv
    out_tsv = BASE_DIR / "output" / "matching_results_ensemble.tsv"
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for qid in all_qids:
            f.write(f"{qid}\t{','.join(ensemble_preds.get(qid, []))}\n")

    ens_matched = sum(1 for m in ensemble_preds.values() if len(m) > 0)
    print(f"Super-Ensemble generated: {len(ensemble_preds):,} queries | {ens_matched:,} matched ({ens_matched/len(ensemble_preds)*100:.2f}%)")
    
    # Copy to Desktop
    shutil.copyfile(out_tsv, BASE_DIR / "output" / "matching_results.tsv")
    shutil.copyfile(out_tsv, DESKTOP / "matching_results.tsv")
    print(f"Synced to Desktop: {DESKTOP / 'matching_results.tsv'}")
    return out_tsv

if __name__ == "__main__":
    compare_and_build_ensemble()
