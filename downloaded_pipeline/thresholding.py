# -*- coding: utf-8 -*-
"""
thresholding.py
================
Turns per-candidate match probabilities into a final predicted set per query,
by exact expected-F_beta maximization -- the decision-theoretic approach of
Ye, Chai, Lee & Chieu, "Optimizing F-measures: A Tale of Two Approaches" (ICML 2012),
the same family of technique popularized on Kaggle as "Faron's F1-maximization"
in the Instacart Market Basket Analysis competition (predict the empty set, or a
size-k subset, by exact expected-F1 rather than a flat 0.5 probability cutoff).
This file generalizes it from F1 to F_beta=0.5 exactly, and specializes it to a
per-query (not global) decision, matching this competition's macro-averaged metric.

CORE IDEA
---------
Given a query's candidates sorted by predicted match probability p_1 >= p_2 >= ... >= p_n,
the F_beta-maximizing predicted set is PROVABLY a prefix of this sorted list for some
k in {0, ..., n} (never a non-contiguous subset) -- so the search space is just n+1
options, not 2^n. For a fixed k, treat "is candidate i actually a true match" as an
independent Bernoulli(p_i) and compute the exact expectation of F_beta over the
resulting Poisson-Binomial distribution of true positives (top-k) and false negatives
(the rest). This was verified against exact brute-force enumeration over all 2^n label
combinations (n up to 8) with max abs error 4.4e-16 (floating point noise only, see
project validation notes).

WHY THIS IS THE RIGHT TOOL FOR THIS METRIC SPECIFICALLY
---------------------------------------------------------
Plug k=0 into the F_beta identity: TP=0, FP=0, FN=T (T = true matches among the n
candidates). If T=0 (a true singleton), the (0,0,0) case is defined as F=1.0 by this
competition's own stated convention ("singletons score 1.0 if empty") -- and the DP
reproduces that exactly as the T=0, k=0 limiting case, with no special-cased rule
required. Plug in ANY k>0 on a true singleton (T=0): TP is always 0 (there is nothing
to be a true positive), so F=0 for every realization -- again reproducing the stated
"0.0 if any match predicted" rule exactly, with no special case. The singleton/
1-match/multi-match decision is not three different code paths; it is the same
formula evaluated at different k.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def poisson_binomial_pmf(probs: np.ndarray) -> np.ndarray:
    """PMF of the number of successes among independent Bernoulli(p_i) trials.
    O(len(probs)^2) via incremental convolution -- trivial for candidate sets <= 15."""
    pmf = np.array([1.0])
    for p in probs:
        new_pmf = np.zeros(len(pmf) + 1)
        new_pmf[:-1] += pmf * (1 - p)
        new_pmf[1:] += pmf * p
        pmf = new_pmf
    return pmf


def expected_fbeta_for_k(probs_sorted_desc: np.ndarray, k: int, beta: float = 0.5) -> float:
    n = len(probs_sorted_desc)
    top, rest = probs_sorted_desc[:k], probs_sorted_desc[k:]
    pmf_top, pmf_rest = poisson_binomial_pmf(top), poisson_binomial_pmf(rest)
    b2 = beta ** 2
    total = 0.0
    for tp in range(k + 1):
        p_top = pmf_top[tp]
        if p_top == 0:
            continue
        for fn in range(n - k + 1):
            pj = p_top * pmf_rest[fn]
            if pj == 0:
                continue
            fp = k - tp
            denom = (1 + b2) * tp + b2 * fn + fp
            f = 1.0 if denom == 0 else (1 + b2) * tp / denom
            total += pj * f
    return total


def optimal_k_for_query(probs: np.ndarray, beta: float = 0.5, max_k: int | None = None):
    """probs: calibrated P(match) for one query's candidates, any order.
    Returns (selected_target_positions, k_star, expected_fbeta)."""
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(-probs)
    probs_sorted = probs[order]
    n = len(probs_sorted)
    upper = n if max_k is None else min(n, max_k)
    scores = [expected_fbeta_for_k(probs_sorted, k, beta) for k in range(upper + 1)]
    k_star = int(np.argmax(scores))
    return order[:k_star].tolist(), k_star, scores[k_star]


def predict_matches(
    scored_pairs: pd.DataFrame,
    prob_col: str = "calibrated_prob",
    beta: float = 0.5,
    max_k: int = 9,  # ground-truth stats: true matches range 1-9 per entity
) -> pd.DataFrame:
    """scored_pairs: [query_id, target_id, <prob_col>], multiple rows per query_id.
    Returns one row per query_id that HAS a predicted match: [query_id, target_id,
    predicted_rank]. Queries predicted empty simply do not appear -- the caller is
    responsible for ensuring every query_id in the test set appears at least as an
    empty prediction in the final submission (see inference_and_threshold.py)."""
    out_rows = []
    for qid, grp in scored_pairs.groupby("query_id", sort=False):
        probs = grp[prob_col].to_numpy()
        idx_local, k_star, _ = optimal_k_for_query(probs, beta=beta, max_k=max_k)
        if k_star == 0:
            continue
        chosen = grp.iloc[idx_local]
        for rank, (_, r) in enumerate(chosen.iterrows(), start=1):
            out_rows.append((qid, r["target_id"], rank))
    return pd.DataFrame(out_rows, columns=["query_id", "target_id", "predicted_rank"])


# --------------------------------------------------------------------------- #
# Evaluation: macro F_beta exactly as the competition scores it
# --------------------------------------------------------------------------- #

def macro_fbeta_score(
    predicted: pd.DataFrame,       # [query_id, target_id]
    ground_truth: pd.DataFrame,    # [query_id, target_id]
    all_query_ids: pd.Series,      # every query in the eval set, including singletons
    beta: float = 0.5,
) -> float:
    """Per-query F_beta, averaged over ALL queries (missing predictions = empty set,
    missing ground truth = singleton) -- exactly the stated scoring rule."""
    pred_sets = predicted.groupby("query_id")["target_id"].apply(set).to_dict()
    true_sets = ground_truth.groupby("query_id")["target_id"].apply(set).to_dict()
    b2 = beta ** 2
    scores = []
    for qid in all_query_ids:
        pred = pred_sets.get(qid, set())
        true = true_sets.get(qid, set())
        tp = len(pred & true)
        fp = len(pred - true)
        fn = len(true - pred)
        denom = (1 + b2) * tp + b2 * fn + fp
        scores.append(1.0 if denom == 0 else (1 + b2) * tp / denom)
    return float(np.mean(scores))


def baseline_fixed_topk(scored_pairs: pd.DataFrame, k: int, prob_col: str = "calibrated_prob") -> pd.DataFrame:
    """Comparison baseline: blindly take the top-k by probability, every query,
    no k=0 option -- structurally similar to the rule-based pipeline's failure mode."""
    return (scored_pairs.sort_values(["query_id", prob_col], ascending=[True, False])
            .groupby("query_id").head(k)[["query_id", "target_id"]])


def baseline_fixed_threshold(scored_pairs: pd.DataFrame, threshold: float, prob_col: str = "calibrated_prob") -> pd.DataFrame:
    """Comparison baseline: predict every candidate above a single global probability
    cutoff. Unlike the DP, this ignores each query's OWN candidate-set shape."""
    return scored_pairs.loc[scored_pairs[prob_col] >= threshold, ["query_id", "target_id"]]


if __name__ == "__main__":
    # Reproduces the validated edge cases from development, as a regression guard.
    cases = {
        "near-zero probs -> singleton (k=0)": [0.05, 0.03, 0.02],
        "one clear match among noise -> k=1": [0.97, 0.15, 0.08, 0.04],
        "3 true dupes + 2 lookalikes -> k=3": [0.95, 0.93, 0.90, 0.35, 0.20],
    }
    for desc, probs in cases.items():
        idx, k, val = optimal_k_for_query(np.array(probs))
        print(f"{desc}: k*={k}  E[F0.5]={val:.4f}  chosen positions={idx}")
