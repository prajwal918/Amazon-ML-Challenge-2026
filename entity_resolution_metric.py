#!/usr/bin/env python3
"""
Verified Macro F0.5 Scorer for Amazon ML Challenge 2026.
Matches official organizers' specification and worked examples exactly.
"""

from typing import Dict, Set


def compute_entity_f05(pred: Set[str], truth: Set[str]) -> float:
    """Compute F0.5 score for a single Source 1 entity.

    - True singleton correctly predicted (both empty): 1.0
    - True singleton falsely matched (truth empty, pred non-empty): 0.0
    - Non-empty truth predicted empty: 0.0
    - Otherwise: F0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    if not truth and not pred:
        return 1.0
    if not truth or not pred:
        return 0.0

    inter = len(pred & truth)
    if inter == 0:
        return 0.0

    precision = inter / len(pred)
    recall = inter / len(truth)

    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0

    return (1.25 * precision * recall) / denom


def evaluate_predictions(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Compute Macro F0.5, Macro Precision, and Macro Recall over all entities."""
    all_s1 = set(ground_truth.keys())
    
    total_f05 = 0.0
    total_prec = 0.0
    total_rec = 0.0
    n = len(all_s1)

    for s1_id in all_s1:
        truth = ground_truth.get(s1_id, set())
        pred = predictions.get(s1_id, set())

        score = compute_entity_f05(pred, truth)
        total_f05 += score

        if not truth and not pred:
            total_prec += 1.0
            total_rec += 1.0
        elif not truth and pred:
            total_prec += 0.0
            total_rec += 1.0  # recalled the empty set: 1, but precision: 0
        elif truth and not pred:
            total_prec += 1.0
            total_rec += 0.0
        else:
            inter = len(pred & truth)
            total_prec += inter / len(pred)
            total_rec += inter / len(truth)

    return {
        "macro_f05": total_f05 / n if n > 0 else 0.0,
        "macro_precision": total_prec / n if n > 0 else 0.0,
        "macro_recall": total_rec / n if n > 0 else 0.0,
        "num_entities": n,
    }


if __name__ == "__main__":
    # Test with official worked example:
    # Pred: [S2-00047, S2-00193, S3-00812]
    # Truth: [S2-00047, S3-00812]
    p = {"S2-00047", "S2-00193", "S3-00812"}
    g = {"S2-00047", "S3-00812"}
    score = compute_entity_f05(p, g)
    print(f"Worked Example Test: Expected ~0.714, Got: {score:.4f}")
    assert round(score, 3) == 0.714, "Unit test failed on worked example!"
    print("Verification passed successfully!")
