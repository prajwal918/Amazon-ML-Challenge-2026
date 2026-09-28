# -*- coding: utf-8 -*-
"""
inference_and_threshold.py
============================
End-to-end batch inference for the 1.7M-query test set:
  1. Stage-1 candidate generation (reuses stage1_blocking.generate_candidates)
  2. Feature computation (reuses feature_engineering.build_feature_matrix)
  3. LightGBM scoring + isotonic calibration (model trained in train_lightgbm.py)
  4. Per-query F0.5-optimal decision (thresholding.predict_matches)
  5. Submission assembly -- CRITICALLY, every query_id is emitted, including those
     predicted empty. A query silently missing from a submission file is easy to
     misread as "not scored" rather than "predicted singleton"; make the empty
     prediction explicit.

Runtime budget (validated during development, single machine, will vary with your
actual data's block-size distribution and core count):
  - DP thresholding step alone: ~400 microseconds/query -> ~11 min for 1.7M queries
    single-core, ~3 min split across 4 cores. This step is not your bottleneck.
  - Stage-1 blocking and feature computation across ~1.7M x 15 =~ 25M pairs are the
    parts to benchmark on a representative sample of YOUR data before committing to
    the full 10M-target run; the chunk sizes in BlockingConfig are tuning knobs for
    the 30GB RAM ceiling, not fixed constants.
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import joblib

from stage1_blocking import generate_candidates, build_corpus_frequency, BlockingConfig
from feature_engineering import build_feature_matrix, FULL_FEATURE_NAMES
from thresholding import predict_matches


def run_inference(
    test_queries: pd.DataFrame,
    target_pool: pd.DataFrame,
    model_bundle_path: str,
    cfg: BlockingConfig = BlockingConfig(),
    beta: float = 0.5,
    max_k: int = 9,
    feature_batch_size: int = 200_000,
) -> pd.DataFrame:
    bundle = joblib.load(model_bundle_path)
    model, calibrator, feature_cols = bundle["model"], bundle["calibrator"], bundle["feature_cols"]

    corpus_freq = build_corpus_frequency(
        pd.concat([test_queries["business_name"], target_pool["business_name"]])
    )
    print(f"[infer] generating candidates for {len(test_queries)} queries "
          f"against {len(target_pool)} targets...")
    candidates = generate_candidates(test_queries, target_pool, cfg, corpus_freq)
    print(f"[infer] {len(candidates)} candidate pairs "
          f"({len(candidates) / len(test_queries):.2f} avg per query)")

    q_cols = test_queries.rename(columns={
        "entity_id": "query_id", "business_name": "q_business_name",
        "business_address": "q_business_address", "country": "q_country",
    })[["query_id", "q_business_name", "q_business_address", "q_country"]]
    t_cols = target_pool.rename(columns={
        "entity_id": "target_id", "business_name": "c_business_name",
        "business_address": "c_business_address", "country": "c_country",
    })[["target_id", "c_business_name", "c_business_address", "c_country"]]
    pairs = candidates.merge(q_cols, on="query_id", how="left").merge(t_cols, on="target_id", how="left")

    scored_chunks = []
    for start in range(0, len(pairs), feature_batch_size):
        chunk = pairs.iloc[start:start + feature_batch_size]
        feats = build_feature_matrix(chunk, corpus_freq=corpus_freq)
        raw = model.predict(feats[feature_cols], num_iteration=model.best_iteration)
        feats["calibrated_prob"] = calibrator.predict(raw)
        scored_chunks.append(feats[["query_id", "target_id", "calibrated_prob"]])
        print(f"[infer] scored {min(start + feature_batch_size, len(pairs))}/{len(pairs)} pairs")
    scored = pd.concat(scored_chunks, ignore_index=True)

    print("[infer] applying F0.5-optimal per-query thresholding...")
    matches = predict_matches(scored, prob_col="calibrated_prob", beta=beta, max_k=max_k)

    # Guarantee every query_id appears, including predicted-empty singletons.
    all_qids = pd.DataFrame({"query_id": test_queries["entity_id"].unique()})
    submission = (all_qids.merge(matches, on="query_id", how="left")
                           .groupby("query_id")["target_id"]
                           .apply(lambda s: [x for x in s if pd.notna(x)])
                           .reset_index())
    n_empty = (submission["target_id"].map(len) == 0).sum()
    print(f"[infer] {n_empty}/{len(submission)} queries predicted as singletons "
          f"({n_empty/len(submission):.4f})")
    return submission


def write_submission(submission: pd.DataFrame, path: str, sep: str = ","):
    """Adjust to your competition's exact required format -- this default writes
    one row per query with a delimiter-joined list of predicted target_ids, empty
    string for predicted singletons."""
    out = submission.copy()
    out["predicted_matches"] = out["target_id"].map(lambda ids: sep.join(ids))
    out[["query_id", "predicted_matches"]].to_csv(path, index=False)
    print(f"[infer] wrote submission -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default=None, help="path to test queries file")
    parser.add_argument("--targets", default=None, help="path to target pool file")
    parser.add_argument("--model", default="/tmp/entity_resolution_model.joblib")
    parser.add_argument("--out", default="/tmp/submission.csv")
    args = parser.parse_args()

    if args.queries and args.targets:
        test_queries = pd.read_csv(args.queries, sep="\t")
        target_pool = pd.read_csv(args.targets, sep="\t")
    else:
        # ---- Synthetic smoke test using the model trained by train_lightgbm.py ----
        rng = np.random.default_rng(2)
        base_names = ["First Food", "Moore Bitwise", "Smart Development", "Network Solutions",
                      "All International", "Riverside Traders", "Golden Gate Logistics"]
        NOISE_FNS = [lambda s: s, lambda s: s.upper(),
                     lambda s: " ".join(reversed(s.split())),
                     lambda s: s.replace("s", "5", 1), lambda s: s + " Pvt Ltd"]
        q_rows, t_rows = [], []
        tid = 0
        for i, base in enumerate(base_names * 8):
            qid = f"S1-{i}"
            addr = rng.integers(1, 999)
            q_rows.append(dict(entity_id=qid, business_name=base,
                                business_address=f"{addr} Main St", country="US"))
            if rng.random() > 0.15:
                for _ in range(rng.integers(1, 3)):
                    t_rows.append(dict(entity_id=f"S2-{tid}", business_name=rng.choice(NOISE_FNS)(base),
                                        business_address=f"{addr} Main St", country="US"))
                    tid += 1
        for _ in range(200):
            t_rows.append(dict(entity_id=f"S3-{tid}",
                                business_name=rng.choice(base_names) + f" Branch{rng.integers(1,50)}",
                                business_address=f"{rng.integers(1,999)} Other Ave", country="US"))
            tid += 1
        test_queries, target_pool = pd.DataFrame(q_rows), pd.DataFrame(t_rows)

    submission = run_inference(test_queries, target_pool, args.model)
    write_submission(submission, args.out)
    print(submission.head(10).to_string())
