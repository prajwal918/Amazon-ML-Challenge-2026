# -*- coding: utf-8 -*-
"""
train_lightgbm.py
==================
Builds the labeled (query, candidate) training table from Stage-1 candidates +
train_ground_truth.tsv, trains a calibrated LightGBM binary classifier, and reports
macro F0.5 (via thresholding.py) against naive top-K / fixed-threshold baselines so
the improvement is visible on your own validation fold before you touch the test set.

Expected input schema (adjust column names to your actual files):
  train_ground_truth.tsv : columns [query_id, target_id]   (one row per true match;
                            a query_id with zero rows is a ground-truth singleton)
  train queries / targets: columns [entity_id, business_name, business_address, country]
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
import joblib

from stage1_blocking import generate_candidates, build_corpus_frequency, BlockingConfig
from feature_engineering import build_feature_matrix, FULL_FEATURE_NAMES
from thresholding import (
    predict_matches, macro_fbeta_score, baseline_fixed_topk, baseline_fixed_threshold,
)


def build_training_table(
    train_queries: pd.DataFrame,
    train_targets: pd.DataFrame,
    ground_truth: pd.DataFrame,
    cfg: BlockingConfig = BlockingConfig(),
) -> pd.DataFrame:
    corpus_freq = build_corpus_frequency(
        pd.concat([train_queries["business_name"], train_targets["business_name"]])
    )
    candidates = generate_candidates(train_queries, train_targets, cfg, corpus_freq)

    # --- Audit Stage 1 recall BEFORE doing anything else. Stage 2 cannot recover a
    # true match Stage 1 never retrieved -- this number is your hard recall ceiling. ---
    gt_pairs = set(zip(ground_truth["query_id"], ground_truth["target_id"]))
    retrieved_pairs = set(zip(candidates["query_id"], candidates["target_id"]))
    recall_at_k = len(gt_pairs & retrieved_pairs) / max(len(gt_pairs), 1)
    print(f"[audit] Stage-1 blocking recall on training ground truth: {recall_at_k:.4f} "
          f"({len(gt_pairs & retrieved_pairs)}/{len(gt_pairs)} true pairs retrieved)")
    if recall_at_k < 0.97:
        print("[audit] WARNING: recall below 0.97 caps your achievable F0.5 regardless of "
              "Stage 2 quality. Widen blocking (more passes, larger top_k, or looser keys) "
              "before investing further in the classifier.")

    missed = pd.DataFrame(list(gt_pairs - retrieved_pairs), columns=["query_id", "target_id"])
    if len(missed):
        missed["blocking_score"] = 0.0  # force-included so the classifier still sees these
        candidates = pd.concat([candidates, missed], ignore_index=True)
        candidates = candidates.drop_duplicates(["query_id", "target_id"])

    q_cols = train_queries.rename(columns={
        "entity_id": "query_id", "business_name": "q_business_name",
        "business_address": "q_business_address", "country": "q_country",
    })[["query_id", "q_business_name", "q_business_address", "q_country"]]
    t_cols = train_targets.rename(columns={
        "entity_id": "target_id", "business_name": "c_business_name",
        "business_address": "c_business_address", "country": "c_country",
    })[["target_id", "c_business_name", "c_business_address", "c_country"]]

    pairs = candidates.merge(q_cols, on="query_id", how="left").merge(t_cols, on="target_id", how="left")
    pairs["label"] = pairs.set_index(["query_id", "target_id"]).index.isin(gt_pairs).astype(int)

    features = build_feature_matrix(pairs, corpus_freq=corpus_freq)
    return features


def train_model(features: pd.DataFrame, feature_cols=FULL_FEATURE_NAMES, n_splits: int = 5, seed: int = 0):
    groups = features["query_id"]
    gkf = GroupKFold(n_splits=n_splits)
    train_idx, val_idx = next(gkf.split(features, groups=groups))
    tr, va = features.iloc[train_idx], features.iloc[val_idx]

    n_pos, n_neg = tr["label"].sum(), (tr["label"] == 0).sum()
    params = dict(
        objective="binary", metric="auc", learning_rate=0.05, num_leaves=63,
        min_data_in_leaf=50, feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
        scale_pos_weight=n_neg / max(n_pos, 1),   # candidates are mostly negatives by construction
        seed=seed, verbose=-1,
    )
    dtrain = lgb.Dataset(tr[feature_cols], label=tr["label"])
    dval = lgb.Dataset(va[feature_cols], label=va["label"], reference=dtrain)
    model = lgb.train(
        params, dtrain, num_boost_round=2000, valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    raw_val_scores = model.predict(va[feature_cols], num_iteration=model.best_iteration)
    # Isotonic calibration: the DP thresholding step assumes p_i are true probabilities,
    # not just monotonic scores -- calibration is a precondition for it, not an extra.
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_val_scores, va["label"])

    va = va.copy()
    va["raw_score"] = raw_val_scores
    va["calibrated_prob"] = calibrator.predict(raw_val_scores)
    return model, calibrator, va, feature_cols


def evaluate_against_baselines(scored_val: pd.DataFrame, ground_truth: pd.DataFrame, beta: float = 0.5):
    all_qids = scored_val["query_id"].unique()
    gt_val = ground_truth[ground_truth["query_id"].isin(all_qids)]

    dp_pred = predict_matches(scored_val, prob_col="calibrated_prob", beta=beta)
    f_dp = macro_fbeta_score(dp_pred, gt_val, pd.Series(all_qids), beta)

    top5_pred = baseline_fixed_topk(scored_val, k=5, prob_col="calibrated_prob")
    f_top5 = macro_fbeta_score(top5_pred, gt_val, pd.Series(all_qids), beta)

    thr_pred = baseline_fixed_threshold(scored_val, threshold=0.5, prob_col="calibrated_prob")
    f_thr = macro_fbeta_score(thr_pred, gt_val, pd.Series(all_qids), beta)

    print(f"[eval] Macro F{beta} -- blind top-5 baseline      : {f_top5:.4f}")
    print(f"[eval] Macro F{beta} -- fixed p>=0.5 baseline      : {f_thr:.4f}")
    print(f"[eval] Macro F{beta} -- DP-optimal per-query (ours): {f_dp:.4f}")
    return {"top5": f_top5, "fixed_threshold": f_thr, "dp_optimal": f_dp}


if __name__ == "__main__":
    # ---- Synthetic end-to-end smoke test (replace with real data loaders) ----
    rng = np.random.default_rng(1)
    base_names = ["First Food", "Moore Bitwise", "Smart Development", "Network Solutions",
                  "All International", "Escobedo Smart of Williamson", "Riverside Traders",
                  "Golden Gate Logistics", "Blue Ridge Consulting", "Sunrise Textiles"]
    NOISE_FNS = [
        lambda s: s,
        lambda s: s.upper(),
        lambda s: " ".join(reversed(s.split())),          # word permutation
        lambda s: s.replace("s", "5", 1).replace("S", "5", 1),  # leetspeak
        lambda s: s + " Pvt Ltd",
        lambda s: s + " LLC",
    ]
    queries, targets, gt = [], [], []
    tid_counter = 0
    for i, base in enumerate(base_names * 15):  # 150 base entities
        qid = f"S1-{i}"
        addr_num = rng.integers(1, 999)
        queries.append(dict(entity_id=qid, business_name=base,
                             business_address=f"{addr_num} Main St", country="US"))
        n_clones = 0 if rng.random() < 0.15 else rng.integers(1, 4)  # ~15% singleton rate
        for _ in range(n_clones):
            noisy_name = rng.choice(NOISE_FNS)(base)
            noisy_addr = f"{addr_num:04d} Main St" if rng.random() < 0.5 else f"{addr_num} Main St"
            tid = f"S2-{tid_counter}"; tid_counter += 1
            targets.append(dict(entity_id=tid, business_name=noisy_name,
                                 business_address=noisy_addr, country="US"))
            gt.append((qid, tid))
    for _ in range(400):  # unrelated distractor pool
        tid = f"S3-{tid_counter}"; tid_counter += 1
        targets.append(dict(entity_id=tid, business_name=rng.choice(base_names) + f" Branch{rng.integers(1,50)}",
                             business_address=f"{rng.integers(1,999)} Other Ave", country="US"))

    train_queries = pd.DataFrame(queries)
    train_targets = pd.DataFrame(targets)
    ground_truth = pd.DataFrame(gt, columns=["query_id", "target_id"])
    print(f"synthetic: {len(train_queries)} queries, {len(train_targets)} targets, "
          f"{len(ground_truth)} true pairs, "
          f"{train_queries['entity_id'].nunique() - ground_truth['query_id'].nunique()} singletons")

    features = build_training_table(train_queries, train_targets, ground_truth)
    model, calibrator, val_scored, feat_cols = train_model(features)
    evaluate_against_baselines(val_scored, ground_truth)

    joblib.dump({"model": model, "calibrator": calibrator, "feature_cols": feat_cols},
                "/tmp/entity_resolution_model.joblib")
    print("saved model bundle -> /tmp/entity_resolution_model.joblib")
