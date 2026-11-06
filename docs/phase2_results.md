# Phase 2 Results — Frozen Protocol v1.0

**Status: core experiment and outputs generated; independent review pending.** No final PDF or course submission was produced. These are historical predictive results on the frozen sample, not evidence that review prevents default. A repository commit was created later at the student's request; no tag was created.

## Freeze and reproducibility record

- Protocol: `docs/evaluation_protocol_v1.0.md`; SHA-256 `03ffa36a02fcdbbf8df8484ffdb84f58e063819899409812cb4e46a87abae418`.
- Split and fold artifact: `results/audit/proposed_splits.json`; SHA-256 `ded744149bd5afead62f3355d773eded3e9be1c8a53a5deabee38e104f42b1ed`.
- Original UCI ZIP SHA-256: `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`.
- Environment: Conda `nussem1`; Python 3.10.18, pandas 2.3.1, NumPy 2.2.6, scikit-learn 1.7.1, xlrd 2.0.1, matplotlib 3.10.5, nbformat 5.10.4, nbclient 0.10.2, pytest 8.4.2; macOS 15.5 arm64.
- 180/180 CV configuration-fold fits succeeded (60 per model); no fit failures or convergence warnings. Recorded CV fit times total 1,155.3 seconds: Logistic Regression 9.0s, Random Forest 975.5s, HistGradientBoosting 170.9s. This is cumulative fit time, not equal CPU budget.
- Targeted automated verification: 10 tests passed, covering frozen hashes/splits, schema and leakage allowlist, shared preprocessing, ECE, calibration separation, paired-bootstrap resampling, locked threshold application, and prediction alignment.
- Final Test: 4,500 rows (995 positive, 3,505 negative). Bootstrap: 2,000 shared-index resamples, seed 2026; all 2,000 had both classes, so none were rejected for AP/ROC-AUC.
- The notebook was clean-kernel executed with `RUN_FULL_EXPERIMENT=False`: it revalidated the frozen artifacts, regenerated reports/figures, and displayed the saved outcomes without fitting another model. The complete CV/test artifacts were already present. A repeated full-search attempt was interrupted during Random Forest CV and did not replace the completed 180-row CV results or fitted artifacts. An earlier finalization stalled in report rendering; the point-by-point plotting loop was replaced by grouped curve plotting. These execution incidents are retained in `results/logs/phase2_execution_incidents.md`.

## Selected configurations

Selection used mean five-fold AP; exact ties would select the earliest grid entry. All models used the same frozen folds and Pipeline preprocessing.

| Model | Selected parameters | Mean CV AP | Fold SD |
| --- | --- | ---: | ---: |
| Logistic Regression | `C=3.0`, `class_weight=None` | 0.54250 | 0.01090 |
| Random Forest | `max_depth=8`, `min_samples_leaf=5`, `max_features=0.7` | 0.55652 | 0.01741 |
| HistGradientBoosting | `learning_rate=0.03`, `max_leaf_nodes=15`, `l2_regularization=0.0` | 0.56058 | 0.01367 |

All CV fold scores, configuration grids, fit seconds, warnings, selected parameters, and fitted model artifacts are stored in `results/cv/` and `results/models/`.

## Final Test discrimination and probability quality

| Model | AP raw / sigmoid | ROC-AUC raw / sigmoid | Brier raw / sigmoid | ECE raw / sigmoid |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.5574 / 0.5574 | 0.7694 / 0.7694 | 0.13528 / 0.13553 | 0.01370 / 0.01538 |
| Random Forest | 0.5750 / 0.5750 | 0.7820 / 0.7820 | 0.13274 / 0.13368 | 0.01315 / 0.01785 |
| HistGradientBoosting | 0.5798 / 0.5798 | 0.7882 / 0.7882 | 0.13200 / 0.13192 | 0.01113 / 0.01060 |

Sigmoid calibration preserved ranking metrics, as expected for a monotonic mapping, but did not improve every probability metric. It worsened both Brier and ECE for Logistic Regression and Random Forest; it slightly improved both for HistGradientBoosting. This is a concrete counterexample to assuming calibration always helps.

### Calibrated model estimates and 95% percentile intervals

| Model | AP | ROC-AUC | Brier | ECE |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.5574 [0.5246, 0.5894] | 0.7694 [0.7521, 0.7869] | 0.13553 [0.12930, 0.14209] | 0.01538 [0.00992, 0.02773] |
| Random Forest | 0.5750 [0.5430, 0.6058] | 0.7820 [0.7646, 0.7987] | 0.13368 [0.12721, 0.14013] | 0.01785 [0.01249, 0.03180] |
| HistGradientBoosting | 0.5798 [0.5460, 0.6115] | 0.7882 [0.7712, 0.8047] | 0.13192 [0.12564, 0.13815] | 0.01060 [0.00913, 0.02474] |

### Paired differences versus Logistic Regression

Each entry is compared model minus Logistic Regression, using identical bootstrap rows. Negative differences favor the compared model for Brier/ECE; positive differences favor it for AP/ROC-AUC.

| Score type | Model | AP difference [95% CI] | ROC-AUC difference [95% CI] | Brier difference [95% CI] | ECE difference [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Raw | Random Forest | +0.01759 [0.00271, 0.03165] | +0.01262 [0.00201, 0.02222] | -0.00254 [-0.00414, -0.00094] | -0.00055 [-0.00908, 0.01068] |
| Raw | HistGradientBoosting | +0.02243 [0.01078, 0.03441] | +0.01877 [0.00934, 0.02809] | -0.00328 [-0.00470, -0.00184] | -0.00257 [-0.00937, 0.01000] |
| Sigmoid | Random Forest | +0.01759 [0.00271, 0.03165] | +0.01262 [0.00201, 0.02222] | -0.00186 [-0.00338, -0.00035] | +0.00247 [-0.00478, 0.01162] |
| Sigmoid | HistGradientBoosting | +0.02243 [0.01078, 0.03441] | +0.01877 [0.00934, 0.02809] | -0.00362 [-0.00508, -0.00219] | -0.00478 [-0.01150, 0.00768] |

The paired ECE intervals include zero for both compared models. Intervals condition on the fitted estimators and this test sample; they exclude model-selection/refit variability, temporal shift, and deployment uncertainty.

## Threshold, cost, and capacity findings

Primary scenario: hypothetical FP cost 1, FN cost 5, maximum 10% flagged on the 3,000-row Threshold Selection subset. Thresholds were locked there and applied unchanged to test.

| Model | Locked threshold | Selection flagged | Selection loss | Test flagged | Test TP / FP / FN | Test loss |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Logistic Regression | 0.567186 | 298 / 3,000 (9.93%) | 2,424 | 416 / 4,500 (9.24%) | 298 / 118 / 697 | 3,603 |
| Random Forest | 0.571389 | 300 / 3,000 (10.00%) | 2,408 | 425 / 4,500 (9.44%) | 312 / 113 / 683 | 3,528 |
| HistGradientBoosting | 0.591527 | 300 / 3,000 (10.00%) | 2,414 | 420 / 4,500 (9.33%) | 309 / 111 / 686 | 3,541 |

Under this single hypothetical cost/capacity rule, Random Forest had the lowest Final Test surrogate loss, while HistGradientBoosting had the highest AP and ROC-AUC. The small loss differences are not monetary savings and do not establish prevented defaults. None of the primary test flag fractions exceeded 10%, but the locked threshold is not a prospective hard-cap guarantee. All nine capacity/cost sensitivity scenarios and their test outcomes are in `results/decisions/cost_capacity_sensitivity.csv`. The separate Top-K results use K = floor(capacity × batch size), with ties resolved by ascending original row position; see `top_k_test_results.csv`.

At test-set Top-K capacity 10% (K=450), TP/FP/FN and hypothetical 1:5 loss were 315/135/680 and 3,535 (Logistic Regression), 327/123/668 and 3,463 (Random Forest), and 326/124/669 and 3,469 (HistGradientBoosting). Top-K is a hard-capacity ranking comparison and is not the locked probability-threshold policy.

## Failure slices and counterexamples

- **No positive PAY code:** 3,042 test records, including 357 observed positives. Under each model's primary locked threshold, none were flagged; all 357 positives were false negatives for all three models. This is a major limitation of the chosen review rule and feature/risk ordering, not a causal conclusion.
- **PAY_0 = 0:** 2,229 records, 285 positives. All three models flagged zero records in this group at their locked primary thresholds, so all 285 positives were false negatives. Slice ranking AP was 0.2431 (Logistic Regression), 0.2428 (Random Forest), and 0.2491 (HistGradientBoosting).
- **Exact primary-profile overlap with Train+CV:** 127 Final Test records (2.82%) exactly matched at least one Train+CV row on the 19 primary features; 57 were positive. For this small overlap group, AP/AUC were 0.5098/0.4821 (Logistic Regression), 0.5794/0.5792 (Random Forest), and 0.5763/0.5916 (HistGradientBoosting). On 4,373 non-overlap records, AUCs were 0.7737, 0.7807, and 0.7874. This is a genuine profile-level weakness/instability signal, but identical profiles do not prove repeated individuals and the 127-row subgroup is limited.
- In the signed bill-to-limit upper tertile (1,461 test rows), Random Forest AP (0.5913) was slightly below Logistic Regression AP (0.5944), while HistGradientBoosting was 0.6003. Subgroup differences are descriptive and no slice-specific tuning was done.

The signed September `BILL_AMT1 / LIMIT_BAL` proxy retained negative values; train-derived tertile cutpoints were 0.062509 and 0.6666. Credit-limit tertile cutpoints were 80,000 and 200,000. Full behavior slices and supplementary demographic aggregate error counts are saved in `results/slices/`. Sparse categories (for example, EDUCATION 4 n=22, EDUCATION 6 n=9, MARRIAGE 0 n=7, and PAY_0 tail groups) are not stable bases for subgroup conclusions. No customer-level narratives or ID-based interpretations were produced.

## Figures and machine-readable evidence

- `results/figures/pr_curves.png`: Final Test PR curves for calibrated scores.
- `results/figures/roc_curves.png`: Final Test ROC curves for calibrated scores.
- `results/figures/reliability_diagrams.png`: raw and calibrated reliability bins with counts in the accompanying CSV.
- `results/figures/calibration_comparison.png`: Brier and ECE raw/calibrated comparisons.
- `results/figures/cost_capacity_sensitivity.png`: locked test loss across planned capacities and cost ratios.
- `results/figures/slice_average_precision.png`: calibrated AP across prespecified behavior slices.
- `results/figures/duplicate_profile_overlap.png`: profile-overlap versus non-overlap test comparison.

Paired predictions are in `results/predictions/final_test_predictions.csv`; bootstrap indices and replicate metrics are in `results/statistics/`; thresholds, costs, and Top-K records are in `results/decisions/`. `results/phase2_execution_report.json` verifies the frozen hashes and output counts.

## Limits and review points

The cohort is historical (2005), and the random split is not temporal validation. End-of-September field availability is an explicit assumption. UCI does not resolve all observed category semantics; states were retained as distinct levels. Identical feature profiles can cross partitions by the approved row-split decision. The high false-negative count in the no-positive-PAY slice matters for the chosen threshold policy. Search counts were equal but compute time was not. Calibration and bootstrap intervals do not solve population shift, model selection uncertainty, or causal intervention questions.

Review the protocol hash, selected-grid records, paired intervals, threshold choices, sparse slices, and execution incident log independently before moving to final reporting/submission work.
