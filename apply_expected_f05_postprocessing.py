#!/usr/bin/env python3
"""
Expected F-0.5 Mathematical Optimizer & Per-Query Stopping Rule
Based on Bayesian Expected F-Beta Maximization:
    F_beta = (1 + beta^2) * T / (beta^2 * N + k)
Where:
    hat_N = sum(p_i)
    hat_p0 = prod(1 - p_i)   [Probability that query is a true singleton]
    For each k in 1..len(candidates):
        score_k = (1 + beta^2) * sum(p_1..p_k) / (beta^2 * hat_N + k)
    If max(score_k) < hat_p0:
        predict 0 matches (preserve singleton)
    Else:
        predict top-k matches
"""

import sys, os
from pathlib import Path

def select_matches_f05(candidate_ids, probs, beta=0.5):
    """
    Returns the subset of candidate_ids to predict as matches (possibly empty),
    maximizing expected F_beta under the model's own probabilities.
    """
    if not candidate_ids or not probs:
        return []
    
    # Sort candidates in descending order of probability
    order = sorted(range(len(probs)), key=lambda i: -probs[i])
    sorted_probs = [float(probs[i]) for i in order]
    sorted_ids = [candidate_ids[i] for i in order]

    hat_N = sum(sorted_probs)
    # hat_p0 is the probability that query has ZERO matches (true singleton)
    hat_p0 = 1.0
    for p in sorted_probs:
        hat_p0 *= max(0.0, 1.0 - p)

    beta2 = beta * beta  # 0.25 for beta=0.5
    best_k = 0
    best_score = hat_p0
    cum = 0.0

    for k in range(1, len(sorted_probs) + 1):
        cum += sorted_probs[k - 1]
        score = (1.0 + beta2) * cum / (beta2 * hat_N + k)
        if score > best_score:
            best_score = score
            best_k = k

    return sorted_ids[:best_k]

if __name__ == "__main__":
    # Quick sanity test
    # Case 1: Weak candidate (p=0.40) -> Expected score for k=0 is 0.60 > score for k=1 (0.45) -> Should pick k=0!
    cands1 = ["T1"]
    probs1 = [0.40]
    res1 = select_matches_f05(cands1, probs1)
    print("Test 1 (Weak match p=0.40):", res1, "(Expected: empty singleton)")
    assert res1 == [], "Test 1 failed!"

    # Case 2: Very strong candidate (p=0.95) -> Should pick k=1!
    cands2 = ["T1", "T2"]
    probs2 = [0.95, 0.15]
    res2 = select_matches_f05(cands2, probs2)
    print("Test 2 (Strong match p=0.95, weak p=0.15):", res2, "(Expected: ['T1'])")
    assert res2 == ["T1"], "Test 2 failed!"

    # Case 3: Two genuine siblings (p1=0.95, p2=0.90) -> Should pick both k=2!
    cands3 = ["T1", "T2", "T3"]
    probs3 = [0.95, 0.90, 0.10]
    res3 = select_matches_f05(cands3, probs3)
    print("Test 3 (Two siblings p=0.95, p=0.90):", res3, "(Expected: ['T1', 'T2'])")
    assert res3 == ["T1", "T2"], "Test 3 failed!"

    print("\nAll Expected F-0.5 mathematical sanity checks passed successfully!")
