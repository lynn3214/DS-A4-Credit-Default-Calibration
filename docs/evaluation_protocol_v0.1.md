# Evaluation protocol v0.1 — provisional

**Not frozen. Human approval required before model implementation or training.** Official assignment requirements are summarized in `DS_A4_TASK_SPEC.md`; the specific dataset, models, split fractions, costs and capacity are student research-design proposals, not instructor mandates.

## Question and boundary

Compare Logistic Regression, Random Forest, and HistGradientBoostingClassifier for next-month default on historical UCI data. The study is an offline risk-review simulation. It estimates associations/prediction under this sample; it does not establish that review or intervention prevents default, reduces losses, or is appropriate for lending decisions.

## Proposed data and features

Target: the source workbook's literal header `default payment next month`, measured as binary 0/1. Candidate primary predictors: `LIMIT_BAL`, `PAY_0`, `PAY_2`–`PAY_6`, six `BILL_AMT` fields, and six `PAY_AMT` fields. Exclude ID and target. Exclude demographic fields from primary models; reserve them for descriptive supplementary error audits. Use natural prevalence. Treat repayment states as categorical (one-hot); fit imputation/scaling within each training fold. Actual frequencies include `EDUCATION` 0/5/6, `MARRIAGE` 0 and `PAY_*` -2/0; UCI's variable description does not resolve these codes. No semantic recoding is approved yet.

## Proposed partition and selection

Stratified 60/15/10/15 Train+CV / Calibration / Threshold Selection / Final Test, seed 42. The generated allocation covers 30,000 rows without overlap and has positive/negative labels in every partition. Five shuffled stratified CV folds in Train+CV have both classes in every training and validation fold, common across all three candidates. Candidate budget proposal: 12 configurations per model, common AP selection metric. Equal candidate counts do not guarantee equal compute budgets. Exact grids, failure handling, preprocessing comparability and any time/resource budget require approval. Exact indices/IDs are persisted in `results/audit/proposed_splits.json`, but remain provisional. Random split supports within-sample interpolation only, not out-of-time generalization. There are no repeated IDs, but 52 duplicate full-predictor groups (108 rows) have 21 groups with conflicting labels; decide whether duplicate profiles may cross partitions before freezing.

After selection, fit each chosen pipeline on Train only; fit sigmoid calibration on the independent Calibration subset without refitting the base model there. Threshold Selection is only for the decision rule. Final Test is reserved for a one-pass assessment. Report AP, ROC-AUC, Brier, 10 equally spaced ECE bins and calibration/reliability summaries; retain raw and calibrated probabilities. Paired test bootstrap proposal: 2,000 shared customer resamples, 95% percentile intervals, checking both classes for ranking metrics. This measures sampling uncertainty for fixed fitted models, not retraining or temporal shift.

## Decision simulation

Proposed surrogate loss: FP=reviewed non-default with cost 1; FN=unreviewed default with cost 5. On Threshold Selection choose a calibrated-score threshold subject to at most 10% selected records, lock its numeric value and apply unchanged to test. State tie handling and infeasibility rules before test inspection. Test review rate can exceed 10% because this is a threshold learned on another sample. Sensitivity proposal: capacities 5/10/20% and FN:FP ratios 2:1, 5:1, 10:1. These are hypothetical assumptions, not measured monetary costs or prevented defaults. Top-K is a separate per-batch policy; strictly monotonic sigmoid calibration does not change within-model ordering.

## Mandatory approval issues and evidence

| Evidence | Concern | Recommendation | Resolve before freeze? |
| --- | --- | --- | --- |
| 52 identical full predictor-vector groups, including 21 with conflicting labels; no repeated IDs | Similar rows may cross partitions; conflicting labels create irreducible ambiguity and row-level uncertainty | Decide whether to keep row split with explicit limitation or group exact full vectors and revalidate stratification | Yes |
| `EDUCATION` 0/5/6, `MARRIAGE` 0, `PAY_*` -2/0 are observed; docs do not define all codes | Semantic assumptions can silently alter representation and model comparisons | Retain codes as distinct categories unless an authoritative interpretation is found; apply same representation policy across models | Yes, before training |
| Source documents monthly fields, not availability timestamps | End-of-September cutoff and availability of September fields are assumptions | Define the simulated decision instant and justify field availability; otherwise narrow claims | Yes |
| 60/15/10/15 stratified split feasible; rare PAY_0 cells in some subsets have 1–6 records | Small-slice metrics will be unstable | Report counts and positives; suppress or qualify tiny-slice estimates | Before freeze |
| Equal proposed 12 configurations/model, AP selection | Same candidate count is not equal compute; unequal grids may advantage one model | Publish grids, fit budget, preprocessing and failure policy before running any CV | Yes |
| Calibration uses 4,500 independent rows; final test is held out | Calibration can still be unstable or worsen probability quality; ECE depends on bins | Lock sigmoid method, bin edges, calibration diagnostics and report raw and calibrated metrics | Yes |
| Proposed 2,000 paired bootstrap resamples | Ranking metrics fail on one-class resamples; intervals do not include refit or temporal uncertainty | Fix shared resampling indices and policy for invalid ranking resamples; describe fixed-fit estimand | Yes, before test evaluation |
| FP/FN ratio 1:5 and 10% capacity are hypothetical; threshold learned on separate 3,000-row sample | These assumptions are not measured costs; test flag rate can exceed selection capacity | Approve primary costs/capacity, tie behavior, infeasible case, sensitivities; label all as simulation | Yes, before threshold selection |
| Human-review capacity is a stipulated percentage, not measured staffing throughput | Capacity may not correspond to actual reviewers or case complexity | Describe as assumed capacity; do not claim operational feasibility | Yes, before decision claims |
| No study data show review changes default outcomes | Predicting future default does not establish that intervention prevents default | Restrict claims to offline risk-review simulation and predictive behavior | Interpretation restriction; causal question cannot be answered here |
| Historical 2005 cohort and random split | No temporal or current-population validity demonstrated | State population/time limits; avoid deployment/fairness guarantee | Interpretation restriction |

The audit script was executed successfully; review its machine-readable output and report before approving. None of these findings authorizes modeling. No protocol freeze or model run has occurred.

All decisions above remain subject to student review. No protocol freeze or model run has occurred.
