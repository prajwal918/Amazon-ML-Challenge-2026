"""
Official Evaluation Metric: Macro-averaged F_0.5 Score for Entity Resolution.
Weights precision 2x over recall: beta = 0.5.
Singletons correctly identified earn 1.0; false merges on singletons earn 0.0.
"""

from typing import Dict, Set


def compute_entity_f_beta(pred_set: Set[str], true_set: Set[str], beta: float = 0.5) -> float:
    """
    Computes F_beta score for a single Source 1 entity.
    """
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    # True singleton
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0

    # Non-empty ground truth with no predictions
    if len(pred_set) == 0:
        return 0.0

    # No true positives found
    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_sq = beta ** 2
    f_score = ((1 + beta_sq) * precision * recall) / ((beta_sq * precision) + recall)
    return f_score


def evaluate_predictions(predictions: Dict[str, Set[str]], ground_truth: Dict[str, Set[str]], beta: float = 0.5) -> float:
    """
    Computes macro-averaged F_beta across all entities in ground truth.
    """
    all_entities = set(ground_truth.keys())
    total_score = 0.0

    for entity_id in all_entities:
        pred_set = predictions.get(entity_id, set())
        true_set = ground_truth.get(entity_id, set())
        score = compute_entity_f_beta(pred_set, true_set, beta=beta)
        total_score += score

    return total_score / len(all_entities) if all_entities else 0.0


if __name__ == "__main__":
    # Worked Example from Problem Statement:
    # Pred: [S2-00047, S2-00193, S3-00812], GT: [S2-00047, S3-00812]
    # Expected F_0.5 = 0.714
    score = compute_entity_f_beta({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(score - 0.714) < 0.001, f"Expected 0.714, got {score:.3f}"
    print(f"Official Worked Example Test: Passed (F_0.5 = {score:.3f})")
