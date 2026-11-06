# DS-A4 — Research Task Specification and Audit-First Plan

**Project root:** `DS-A4/` (initially empty)  
**Course:** UCAS Data Science — DS-A4 Model Comparison and Calibration Lab  
**Status:** Planning / dataset audit; **evaluation protocol v0.1 is NOT frozen**  
**Reference date:** 2026-11-06 (UTC+08:00)

## 1. Authority and scope

### 1.1 Official assignment requirements (instructor-provided document)

Source: `DS-A4_Model_Comparison_and_Calibration_Lab_English.md` supplied by the student; authoritative course website: <https://tjluo-ucas.github.io/ds/>. SPAIQ task is at <https://learn.spaiq.ai/>; any later platform-specific instructions must be checked by the student.

- Release: **2026-11-04 16:30** (Asia/Shanghai); deadline: **2026-11-11 16:30** (Asia/Shanghai); maximum: **5 points**.
- Compare **at least three models**, including a strong simple baseline.
- Use **identical folds, budget and metrics** in paired comparisons.
- Report **PR/ROC, calibration, Brier/ECE and intervals**.
- Analyze **critical slices and failure types**.
- Choose a threshold from **cost and human capacity** and perform sensitivity analysis.
- Deliver: frozen evaluation protocol; paired predictions/statistics; calibration and cost curves; error slices and counterexample; threshold decision record.
- Submit a **150–300-word result summary** and stable **GitHub URL with commit/tag** through the matching SPAIQ task. Keep runnable notebook, PDF, source, raw results and AI-use log in the versioned repository. Do not fabricate runs or results. Document seeds, dependencies, data version, environment, negative results, AI use and independent verification.
- Grading weights: problem/context/method **1.25**, execution/artifacts/reproducibility **1.50**, analysis/counterexample/limits **1.25**, communication/reflection/AI verification **1.00**.

**Important:** Any additional Google Drive instructions must be checked against the current official course submission guide and SPAIQ task before final submission. Do not silently promote recommendations to official requirements.

### 1.2 Selected project (student choice, NOT course-mandated)

**Working title:** *Discrimination, Calibration, and Capacity-Constrained Risk Review: A Paired Comparison of Credit Default Models*.

**Dataset:** UCI Machine Learning Repository, *Default of Credit Card Clients*, Dataset ID 350, DOI <https://doi.org/10.24432/C55S3H>, <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>.

**Study type:** historical, offline, binary probability estimation; **not** a live lending, underwriting, collections, or intervention system.

**Research question:** Under a frozen evaluation protocol, do nonlinear models outperform a strong logistic-regression baseline in discrimination and probability quality for next-month default, and how do calibrated probabilities affect decisions under assumed error costs and limited human-review capacity?

## 2. Research questions and boundaries

- **RQ1 — Discrimination:** Compare average precision (AP) and ROC-AUC across three models on identical held-out observations; quantify paired differences and uncertainty.
- **RQ2 — Probability quality:** Compare raw and independently sigmoid-calibrated probabilities using Brier score, ECE and reliability curves. Better AP/ROC-AUC does **not** imply better calibration.
- **RQ3 — Failure analysis:** Investigate false positives/false negatives and prespecified behavior-based slices; report counterexamples and negative results honestly.
- **RQ4 — Decision:** Pick a risk-review threshold using a separate threshold-selection subset and explicitly hypothetical costs/capacity; evaluate the locked rule on test; vary assumptions.

**Interpretation limits:** Label is next-month default, not preventable default. No data establish that manual review prevents loss. Estimates apply to this historical dataset and comparable sampling conditions, **not** current borrowers or an operational fairness/performance guarantee. A calibrated model on this dataset may fail under base-rate or temporal shift.

## 3. Source facts vs assumptions

Official UCI description states approximately **30,000 rows**, **23 predictors** and binary next-month default target, with past repayment, billing and payment history from **April–September 2005**. These are **expected from source documentation, not locally verified**. `PAY_0` is the September repayment status; there is no `PAY_1`. Source-reported label counts commonly cited are 6,636 positive and 23,364 negative; verify from downloaded bytes.

The dataset has an ID column, four demographic predictors (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`), and nineteen historical financial predictors: `LIMIT_BAL`, `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6`, `BILL_AMT1`…`BILL_AMT6`, `PAY_AMT1`…`PAY_AMT6`.

**Do not assume** UCI's “no missing values” statement proves clean semantic coding. Audit `EDUCATION` 0/5/6, `MARRIAGE` 0, `PAY_*` -2/-1/0 and any other unexpected values. Preserve unrecognized codes until reviewed; never invent meanings.

Assume an end-of-September prediction cutoff **only as a declared methodological assumption**. The source does not establish operational data-availability timestamps. `ID` is provenance only, never a predictor. Inspect ID uniqueness, exact duplicates, feature duplicates and label conflicts before setting a duplicate policy. A cross-sectional random split does **not** validate out-of-time generalization.

## 4. Proposed evaluation protocol v0.1 — PROVISIONAL

**DO NOT FREEZE OR TRAIN until audit approval.** All numbers below are student/assistant proposals, not course requirements.

### 4.1 Samples, target, features

- One observation per customer record, pending ID and duplicate audit.
- Target `default.payment.next.month`: 1 = default next month, 0 = no default next month.
- Primary model feature allowlist (19): `LIMIT_BAL`; `PAY_0`, `PAY_2`–`PAY_6`; `BILL_AMT1`–`BILL_AMT6`; `PAY_AMT1`–`PAY_AMT6`.
- Exclude `ID`, target, and `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` from primary models. Retain demographic fields **only for carefully aggregated supplementary error audit**; report small groups cautiously.
- Keep natural target prevalence; **no** primary oversampling/undersampling. No test-aware feature selection, preprocessing, or tuning.
- Handle `PAY_*` as categorical states with one-hot encoding; numeric columns use fixed imputation and scaling rules fitted within train folds. Review whether treating unknown state codes as separate categories is appropriate after actual audit. Uniform information/feature policy across models.

### 4.2 Partitions and model selection

- Proposed deterministic **60/15/10/15** stratified split: Train+CV / Calibration / Threshold Selection / Final Test (for 30,000 records: 18,000 / 4,500 / 3,000 / 4,500).
- Seed proposal: **42**; persist explicit row IDs and split indices. No overlap. Train-only 5-fold Stratified CV, **same fold indices for every model**.
- Candidate models: **Logistic Regression**, **Random Forest**, **HistGradientBoostingClassifier**.
- Equal declared candidate budget (proposed 12 configurations/model × identical 5 folds); fixed selection metric **AP**; report all metrics and compute time. Equal candidate counts are a *proxy* for comparable search budget, not equal CPU time. Decide exact grids and failure-handling policy **before training**.
- All trained preprocessing is inside a Pipeline. Keep model fitting strictly separated from calibration/threshold/test.

### 4.3 Calibration and final evaluation

- Fit each selected model using Train only, then fit **the same sigmoid/Platt calibration technique** using independent Calibration records, without refitting the model on Calibration records.
- Preserve **both raw and calibrated** predicted probabilities for all three models; calibration may worsen results.
- Evaluate AP, ROC-AUC, Brier score, ECE (prespecified 10 equally spaced probability bins), PR/ROC/reliability curves; show bin sizes. Define AP with `average_precision_score`, not trapezoidal PR AUC.
- Test predictions must be row-aligned and saved, with true label, split ID and model ID. Final test is for **one-pass evaluation**, not selection.
- Use **2,000 paired bootstrap resamples of test customers**, identical indices for each model and paired difference; report 95% percentile CIs and effect sizes. This quantifies test-sampling uncertainty for fixed fitted models, **not** retraining or temporal-shift uncertainty. Check each bootstrap resample has both classes for ranking metrics.

### 4.4 Slices and error/counterexample analysis

Prespecify: recent repayment state (`PAY_0`), history of delinquency (any `PAY_*` > 0, after verifying coding), credit-limit tiers from **train-only** cutoffs, and historical balance/limit burden tiers with explicitly specified treatment of zero/negative values. Aggregate demographic audits may be supplementary, never actionable differentiated credit decisions. Report counts and positives per slice; avoid unstable metrics on tiny subgroups. Do not manufacture a counterexample; report empirically observed failures, contradicted rankings, or honest limits.

### 4.5 Capacity- and cost-sensitive decision

**Offline review simulation only.** Let `flag = 1` imply a customer record is sent for hypothetical risk review. Define `FP` = reviewed but observed non-default and `FN` = not reviewed but observed default. These are **surrogate decision costs**, not measured monetary gains or avoided defaults.

- Proposed primary cost ratio `C_FP=1`, `C_FN=5`, objective `L(t)=C_FP*FP(t)+C_FN*FN(t)`.
- Proposed primary capacity: **at most 10% of threshold-selection records flagged**. Choose the eligible threshold on the **Threshold Selection** split using calibrated probabilities; tie/threshold conventions and infeasible cases must be defined pre-test.
- Lock the **numeric threshold** and apply it unchanged on test. Report observed test flag rate, which can exceed 10% because the capacity was estimated on a different sample; do not claim a hard prospective cap from this rule.
- Sensitivity: capacity limits **5%, 10%, 20%** and cost ratios **2:1, 5:1, 10:1**, planned before test inspection. Report distinct operational policy where desired: **Top-K** per-batch ranking (hard capacity), explaining that strictly monotonic sigmoid calibration should not change ordering and hence cannot improve within-model Top-K selection on the same set. Do not conflate threshold with Top-K.
- If comparing cross-model decisions, fit each model's threshold on the same selection split; do not cherry-pick on test.

## 5. Data audit gate — first Codex phase ONLY

The first phase is **audit and minimal repository bootstrap**, not modeling. It must establish:

1. **Source and integrity:** Retrieve original file from the official UCI origin, record URL/response provenance, raw filename, byte count, SHA-256, retrieval time, license; avoid unofficial mirrors unless explicitly documented and approved.
2. **Physical schema:** Read original spreadsheet including extra header rows if present; verify row count, column names/types, `PAY_0` and no `PAY_1`, target/ID mapping, and expected 23 predictors.
3. **Target:** Verify binary labels, prevalence, counts and any missing/unexpected values.
4. **Feature integrity:** Numeric missing/nonfinite counts, value ranges, and full frequency tables for categorical codes; flag ambiguous codes instead of fixing them silently.
5. **Record integrity:** ID missing/duplicate counts, exact row duplicates, duplicate predictive rows and conflicting labels. Never drop rows without a documented decision.
6. **Data timing and leakage:** Verify named columns against allowlist, no target-derived transformations, discuss cutoff assumptions and absence of temporal external validation.
7. **Split feasibility:** Propose deterministic 60/15/10/15 **indices**; check disjointness and class support; report subgroup sample sizes. Do not reveal test model scores or train models.
8. **Practicality:** Check required Python reader/install versions, local file size and simple read time. Document runtime environment; do not make unsubstantiated hardware/runtime claims.

**Gate decision:** `GO` (protocol can be frozen after human approval), `REVISE` (explicit decisions needed), or `NO-GO` (fundamental issues). No outcome-performance thresholds are allowed as gate criteria.

## 6. First-phase requested repository structure

Codex is authorized to create directories and files **inside the existing empty `DS-A4/` root only**. A suggested minimal structure:

```text
DS-A4/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   └── raw/                      # original verified input; license/size respected
├── scripts/
│   └── audit_dataset.py         # read-only deterministic checks; no training
├── docs/
│   ├── dataset_audit.md         # actual observed findings, clearly sourced
│   ├── evaluation_protocol_v0.1.md # provisional, awaiting approval
│   └── ai_use_log.md            # disclose tool/task, accepted/rejected, checks
└── results/
    └── audit/                    # machine-readable audit evidence
```

This is **a proposal**: prioritize minimal, legible structure over unnecessary scaffolding. `data/raw` must never be silently modified; include external provenance/download instructions and decide Git tracking appropriately. Do not generate notebook, PDF, model files, synthetic results, or final report in phase one.

## 7. Completion contract for phase one

- Code and audit **actually run** against downloaded UCI original; recorded command and exit status, no false claims.
- Audit produces human-readable findings and machine-readable evidence (e.g., JSON/CSV), and includes source file hash, data/target/schema counts, category-frequency tables, duplicate diagnostics, proposed split allocation and unresolved questions.
- Data preprocessing choices, split definitions and feature allowlist are documented but **not yet irreversible/frozen**.
- README describes official task, study scope, reproducible audit command and what has **not** been performed.
- AI-use log records actual first-phase AI assistance, human-verification checklist and any unverified items. **Do not claim the student independently verified work before they do.**
- No model training, calibration, final test evaluation, interpretation of nonexistent results, commits or tags unless separately authorized.
- End with a concise `GO/REVISE/NO-GO` recommendation and exact unresolved decisions for the student/GPT review.

## 8. Later phases (not authorized by first Codex prompt)

1. Human/GPT review actual audit evidence; finalize **Frozen Protocol v1.0** and split rules.
2. Implement models, paired CV, original raw predictions and diagnostics.
3. Implement independent calibration, bootstrap CIs, slices, cost/capacity curves and threshold decision record.
4. Produce runnable notebook + PDF, genuine negative results/counterexample, complete README/AI-use log, clean-kernel verification.
5. Student independently reviews and commits/tags exact results; submit stable repository URL and 150–300-word summary to SPAIQ, checking current platform requirements.

## 9. Sources

- Instructor-provided `DS-A4_Model_Comparison_and_Calibration_Lab_English.md` (teacher assignment source; compare with SPAIQ when accessible).
- Official UCAS DS course: <https://tjluo-ucas.github.io/ds/>.
- UCI dataset: <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>; DOI <https://doi.org/10.24432/C55S3H>.
- scikit-learn documentation for `StratifiedKFold`, `CalibratedClassifierCV`, `average_precision_score` and `brier_score_loss`: <https://scikit-learn.org/stable/>.

**Audit disclaimer:** This specification is based on the previously reviewed instructor document and publicly described dataset. It is not evidence that the source file has been downloaded, row-level facts verified, or experiments run. Record every change from this v0.1 with rationale before final protocol freeze.
