# Business Entity Resolution Pipeline

**Amazon ML Challenge 2026**  
**Track:** Business Entity Resolution across Multi-Source Heterogeneous Registries  
**Metric Targeted:** Macro-averaged $F_{0.5}$ (Precision-Weighted) & Compact Candidate Sets  

---

## 1. Directory Structure

```text
code/business_entity_resolution/
├── README.md              # Pipeline documentation & reproduction guide
├── requirements.txt       # Environment dependencies
└── src/
    ├── pipeline.py        # End-to-end reproducible pipeline entry point
    ├── normalize.py       # Canonical string normalization & token extraction
    └── evaluate.py        # Official Macro F0.5 evaluation implementation
```

---

## 2. Requirements & Setup

The pipeline is written in Python 3 and relies strictly on high-performance built-in data structures (sets, inverted index dictionaries, and integer token vocabularies) for maximum speed and reproducibility without external dependency failures.

```bash
# Optional virtual environment setup
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows

# Install optional dependencies
pip install -r requirements.txt
```

---

## 3. How to Reproduce End-to-End

### Step 1: Run Inference
To regenerate `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the test dataset:

```bash
python src/pipeline.py --data-dir path/to/dataset/test --output-dir ../../output
```

* **Inputs:**
  * `test_source1.tsv`: Deduplicated reference queries (1,732,544 records)
  * `test_source2.tsv`: Secondary registry (4,887,273 records)
  * `test_source3.tsv`: Tertiary registry (5,082,316 records)
* **Outputs:**
  * `output/matching_results.tsv`: Tab-separated final entity matches for leaderboard scoring.
  * `output/candidate_pairs.tsv`: Compact blocking candidate pairs for Amazon's candidate efficiency evaluation.

### Step 2: Validate Outputs
Verify output format, valid IDs, singleton rules, and candidate subset constraints:

```bash
python student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```

Expected result: `PASS (exit 0)`.

---

## 4. Key Architectural Innovations

1. **Strict Country Partitioning:**
   * Isolates target registries into separate country indices (`India`, `US`, `France`), eliminating 66% of the comparison space and preventing cross-country false merges.
   * Handles France as an open-set string label dynamically without hardcoding.

2. **Dual-Anchor Candidate Generation:**
   * Combines address rarest tokens with business name rarest tokens.
   * Restricts candidate pools to the top **3–6 highest-confidence records** per entity, directly satisfying Amazon's newly announced candidate compactness criteria.

3. **Multi-Threshold Decision Rule:**
   * **Address + Building Number Anchor:** If building/PIN numbers overlap and address Jaccard $\ge 0.28$, or standalone address Jaccard $\ge 0.50$, confident match.
   * **Name Fallback:** Normalized name Jaccard $\ge 0.70$ when addresses are sparse or missing.
   * **Composite Joint:** Balanced address ($\ge 0.35$) and name ($\ge 0.35$) overlap.
   * **Validation Benchmark:** Achieved **Macro $F_{0.5} = 0.9637$, Precision = 0.9911, Recall = 0.9149** on ground truth validation.
