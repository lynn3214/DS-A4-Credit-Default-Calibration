# Evaluation Protocol v1.0 — Frozen Before Training

**Status: frozen 2026-11-06, before any Phase 2 model fit.** This protocol is the binding analysis plan for the Phase 2 run. The split artifact and this document must not be edited after training starts. Any scientific correction requires a pause, written rationale, and review before a separately versioned protocol is used.

## 1. Research question and scope

On the historical UCI Default of Credit Card Clients cohort, compare Logistic Regression, Random Forest, and HistGradientBoostingClassifier for discrimination and probability quality when estimating next-month default. Assess how calibrated probabilities affect a hypothetical, capacity-constrained risk-review simulation. The study is predictive and offline. It cannot establish that review causes defaults to be prevented, that any cost unit is monetary, or that the results apply to current borrowers or lending decisions.

## 2. Dataset, unit, target, and cutoff

- Dataset: UCI Machine Learning Repository, *Default of Credit Card Clients*, dataset 350, DOI [10.24432/C55S3H](https://doi.org/10.24432/C55S3H), CC BY 4.0.
- Original source archive: `data/raw/uci_default_credit_card_350.zip`; 5,539,494 bytes; SHA-256 `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`.
- Unit: one original customer record. Retain all rows. Do not delete, merge, relabel, or group records because of duplicate feature profiles.
- Target: literal source column `default payment next month`; positive class 1. Natural prevalence is retained; no resampling.
- Simulated prediction instant: end of September 2005, after April–September account summaries are assumed available. This timestamp and availability assumption are not independently verified from operational source records.

## 3. Feature allowlist and preprocessing

Use exactly these 19 predictive fields: `LIMIT_BAL`; `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6`; `BILL_AMT1`–`BILL_AMT6`; and `PAY_AMT1`–`PAY_AMT6`. Exclude `ID`, target, `SEX`, `EDUCATION`, `MARRIAGE`, and `AGE` from every primary model. Demographics are reserved for aggregate supplementary error auditing only.

Treat the six PAY fields as categorical, including observed codes -2 and 0 as distinct states; assign no undocumented meanings. Use `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` fitted within each Pipeline. Use median imputation followed by `StandardScaler` for the 13 numeric predictors, fitted within each Pipeline. Use most-frequent imputation for PAY fields before one-hot encoding. The audit found no missing values; imputation is a fixed safeguard, not a data-driven choice. The same ColumnTransformer, source rows, represented information, and dense numeric output are used for every candidate. An unseen fold or application category maps to an all-zero one-hot block for that field. No feature selection or target-derived transformations are used.

## 4. Frozen sample split and CV folds

Use the unchanged `results/audit/proposed_splits.json`, seed 42, stratified row split: Train+CV 18,000; Calibration 4,500; Threshold Selection 3,000; Final Test 4,500. SHA-256 of the exact JSON artifact: `ded744149bd5afead62f3355d773eded3e9be1c8a53a5deabee38e104f42b1ed`. The artifact contains all full-data zero-based row indices, aligned IDs, and fold assignments. It is the authoritative exact split and fold definition.

Use its five stored CV train/validation index pairs exactly. Fold indices are positions local to the saved Train+CV ordering. No repartitioning, shuffling, regeneration, or fold reassignment is allowed. All configurations and models use the same five folds. Train/validation folds both contain both target classes. The source has no repeated IDs; 19-feature profile duplicates are retained, including profiles across partitions. Their overlap is diagnostic only and must not affect training or selection.

## 5. Models, search grids, and resource budget

Exactly three models, each with 12 configurations × 5 folds = 60 planned fits. Total planned CV evaluations: 180. Candidate configurations are the Cartesian products below, in listed order. Use one worker (`n_jobs=1`) to avoid uncontrolled parallelism and memory duplication. Record elapsed fit time for every fold and configuration; equal counts do not imply equal CPU time.

1. **Logistic Regression:** Pipeline above plus `LogisticRegression(solver="lbfgs", max_iter=5000, random_state=42)`. Grid: `C` ∈ [0.01, 0.03, 0.1, 0.3, 1.0, 3.0] × `class_weight` ∈ [None, `"balanced"`].
2. **Random Forest:** same preprocessing plus `RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)`. Grid: `max_depth` ∈ [8, 16, None] × `min_samples_leaf` ∈ [1, 5] × `max_features` ∈ [`"sqrt"`, 0.7].
3. **Histogram Gradient Boosting:** same preprocessing plus `HistGradientBoostingClassifier(max_iter=150, random_state=42, early_stopping=False)`. Grid: `learning_rate` ∈ [0.03, 0.1, 0.2] × `max_leaf_nodes` ∈ [15, 31] × `l2_regularization` ∈ [0.0, 1.0].

Use `average_precision_score` on each held-out fold as the sole selection metric. Select the configuration with greatest arithmetic mean of its five fold AP values; exact ties select the earliest configuration in the listed Cartesian-product order. Report fold AP, mean and standard deviation, all parameters, fit time, and warnings. Do not select on test or on a secondary metric.

Log and retain each fit failure. A configuration with any failed fold is incomplete and ineligible for selection; do not replace or retry it with changed settings. Capture all `ConvergenceWarning` instances; do not suppress them. If a selected fit has convergence warnings, report them. If no complete configuration exists for any model, stop before calibration and final-test evaluation and report the blocker. No grid, budget, estimator, feature, or parameter may be changed after observing outcomes.

After selection, refit each selected full Pipeline on Train+CV only. Save fitted base-estimator artifacts. This refit is not calibrated on the same data used to train it.

## 6. Calibration

For each selected fitted base Pipeline, create `CalibratedClassifierCV(estimator=FrozenEstimator(fitted_pipeline), method="sigmoid")`, then fit this calibration wrapper on the independent Calibration subset only. The frozen-estimator wrapper must leave the base Pipeline fitted only on Train+CV. Verify before/after that the base estimator's learned state is unchanged. Do not refit the base on Calibration, Threshold Selection, or Final Test. Use the same procedure for all models. Preserve raw base probabilities and sigmoid-calibrated probabilities. Calibration may worsen metrics.

## 7. Metrics and undefined values

Report raw and calibrated probabilities for all three models.

- Discrimination: non-interpolated average precision (`average_precision_score`) and ROC-AUC (`roc_auc_score`), plus PR and ROC curves.
- Probability quality: Brier score and ECE. ECE uses 10 equal-width bins [0, .1, …, 1]; bins are left-closed/right-open except the final bin includes 1. ECE is the sum over nonempty bins of bin fraction × absolute difference between mean score and observed positive rate. Save every bin's count, mean score, and positive rate. Empty bins have zero weight.
- Threshold results: TP, FP, TN, FN, precision, recall, FPR, fraction flagged, and hypothetical loss.

Every undefined metric is JSON `null`/CSV empty with an explicit reason. Do not substitute zero. AUC/AP are undefined for a one-class evaluation sample; precision is undefined when no rows are flagged; recall is undefined without positives; FPR is undefined without negatives. ECE/Brier require nonempty aligned predictions and binary labels.

## 8. Paired bootstrap

After model selection, calibration, and threshold decisions are locked, evaluate Final Test once. Generate 2,000 bootstrap index vectors of length 4,500 with replacement from Final Test positions, using `numpy.random.default_rng(2026)`. Apply each identical vector to every model and all their outputs. For AP and ROC-AUC, reject one-class resamples and report valid replicate count; calculate percentile 2.5/97.5 endpoints from valid replicates. Brier and ECE are computed for all resamples. Report model-specific bootstrap intervals and paired differences versus Logistic Regression for AP, ROC-AUC, Brier, and ECE using replicate-wise differences and percentile intervals. Negative differences favor the compared model for AP/AUC; positive differences favor it for Brier/ECE. Save replicate-level values and indices or deterministic seed plus sampled indices. These intervals condition on the fitted models and do not quantify model refit/selection uncertainty, temporal shift, or deployment uncertainty.

## 9. Prespecified slices and error analysis

Slice definitions are fixed independently of Final Test model results:

1. **Recent repayment status:** `PAY_0` groups `-2`, `-1`, `0`, and `>=1`; combine small positive delinquency codes as `>=1` without assigning meanings to -2/0. Report label/error counts and metrics only when the slice has both classes and adequate support; always disclose counts.
2. **Any positive historical delay code:** true if any of `PAY_0`, `PAY_2`–`PAY_6` is greater than 0; false otherwise. This is a literal code-based rule, not an asserted interpretation of other repayment codes.
3. **Credit limit:** lower/middle/upper thirds using the 1/3 and 2/3 quantiles of Train+CV `LIMIT_BAL`, calculated once and saved before test analysis. Ties at cutpoints remain in the lower adjacent bin (`pandas.cut` right-closed bins, lowest included). Save exact numeric boundaries.
4. **Optional burden proxy:** signed September bill-to-limit ratio `BILL_AMT1 / LIMIT_BAL`; denominator is positive in all records. Keep negative values unchanged and call it a descriptive proxy, not utilization. Use Train+CV 1/3 and 2/3 quantile boundaries, saved before test analysis. Report all counts; if cutpoints or interpretation prove invalid, omit this optional slice with a logged reason rather than altering definitions after viewing model results.

For each selected model, report slice AP, ROC-AUC, Brier, ECE, TP/FP/TN/FN at the locked primary threshold, and positive/negative counts. If either class is absent, report ranking metrics null with reason. Avoid claims from sparse groups. Supplementary demographic auditing may report aggregated class and error counts for `SEX`, `EDUCATION`, `MARRIAGE`, and age bands; no demographic predictor enters primary models and no customer-level ID narratives are permitted. Report an observed negative result, limitation, or counterexample; never invent one.

## 10. Threshold and cost/capacity decision

On Threshold Selection only, for each calibrated model separately, minimize `FP + 5*FN` subject to flagged fraction ≤ 10%. Candidates are each distinct observed calibrated score plus a no-selection candidate strictly above the maximum. Prediction rule is `p >= threshold`. Reject candidates over capacity. Among equal-loss choices, choose fewer flagged records; if still tied, choose the higher threshold. Save the numeric threshold, selection confusion counts, loss, and fraction. Lock these before test inspection and apply exactly unchanged to Final Test; test capacity may exceed 10% and must be reported without adjustment.

Run all 9 sensitivity scenarios: capacities [5%, 10%, 20%] × FN:FP ratios [2:1, 5:1, 10:1], with loss `FP + ratio*FN`, same threshold candidate and tie rules; choose using Threshold Selection only, then apply unchanged to Final Test. Report every scenario.

Also report Top-K as a separate hard-capacity ranking policy, with `K=floor(capacity × partition_size)` and descending calibrated probability. Ties at the K boundary are resolved by ascending original zero-based dataset row index. Report K, boundary tie count, achieved fraction, and test outcomes. A strictly monotonic sigmoid transform cannot change ordering except numerical ties; it cannot improve within-model Top-K membership on the same records. Do not call a threshold a hard capacity guarantee.

## 11. Duplicate-profile diagnostic

Using the exact 19 primary feature values (no scaling or one-hot transformation for identity matching), quantify exact profile overlap with Train+CV in Calibration, Threshold Selection, and Final Test. Report number and fraction matching any Train+CV profile. On Final Test, compare calibrated model metrics and threshold error counts for overlap versus non-overlap groups with sample sizes and class support. This is a descriptive diagnostic only; do not change split, training, or model choice. Identical observed profiles do not prove duplicate individuals.

## 12. Computation failures and outputs

Use the pinned dependencies recorded in `requirements.txt` and run in `nussem1`. All random states are fixed as stated. Record actual commands, elapsed time, warnings, and failures. Never silently replace undefined values or failed fits. A failure that prevents the frozen analysis is a reported blocker; do not deviate to obtain a result.

Machine-readable outputs under `results/` include CV fold/configuration records, selected parameters, fitted models, row-aligned raw/calibrated predictions, bootstrap replicates and summaries, reliability bins, slice/error tables, threshold decisions, sensitivity scenarios, duplicate overlap, and figures. The notebook is a narrative and executable entry point synchronized to these outputs. No model or decision result may be presented as causal savings or default prevention.

## 13. Limitations and prohibited interpretations

This is a single historical 2005 cohort with a random row split. It provides no temporal validation, current population validation, causal evidence that review prevents default, or operational fairness guarantee. Undocumented category semantics remain unresolved. Identical feature profiles across partitions may inflate apparent profile-level generalization. Equal configuration counts do not equal CPU budgets. Calibration and bootstrap estimates are conditional on this sample/fitted models. Cost units and reviewer capacity are hypothetical. Results cannot justify lending, underwriting, collections, or real-world intervention decisions.

## Freeze record

- Frozen protocol file: `docs/evaluation_protocol_v1.0.md`.
- Frozen split artifact SHA-256: `ded744149bd5afead62f3355d773eded3e9be1c8a53a5deabee38e104f42b1ed`.
- Protocol SHA-256: recorded in `results/protocol_v1.0_freeze.json` immediately after writing this file and before the first model fit; that recorded value is immutable for this run.
- No model fitting had occurred when this protocol was drafted.
