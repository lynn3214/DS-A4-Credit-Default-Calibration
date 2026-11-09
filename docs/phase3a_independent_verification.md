# Phase 3A-1: Independent Numerical Verification

## Scope and verdict

The saved Phase 2 predictions and evidence were independently checked without fitting models, refitting calibrators, searching thresholds, or changing original results. Initial verdict: **FAIL**, because the Phase 2 narrative misstated the `PAY_0 == 0` true-positive counts. Phase 3A-1b corrected that narrative and updated the verifier's expected TP vector. The post-correction rerun outcome is recorded below only after execution.

## Source artifacts and schemas

| Evidence | Path | Relevant fields |
|---|---|---|
| Original UCI workbook | `data/raw/uci_default_credit_card_350.zip` (member `default of credit card clients.xls`) | `ID`, `default payment next month`, and 19 primary feature fields |
| Frozen split | `results/audit/proposed_splits.json` | zero-based `final_test` and `train_cv` row positions |
| Saved Final Test predictions | `results/predictions/final_test_predictions.csv` | `row_position`, `ID`, `y_true`, `split`; six model/probability columns named `{model}__{raw|sigmoid}` |
| Locked thresholds | `results/decisions/locked_decisions.json` | primary model thresholds and 27 sensitivity scenario thresholds |
| Reported Final Test metrics/decisions | `results/statistics/final_test_metrics.csv` | model, probability type, metric values, threshold, confusion counts and loss |
| Reported sensitivity results | `results/decisions/cost_capacity_sensitivity.csv` | scenario keys and test confusion/loss fields |
| Paired bootstrap draws | `results/statistics/bootstrap_indices.npz` | `indices` (2000 × 4500), `seed`, `test_row_positions` |
| Saved bootstrap replicates | `results/statistics/bootstrap_replicates.csv` | replicate/model/probability type, four metrics, ranking validity |
| Saved intervals | `results/statistics/bootstrap_intervals.csv` | point estimates, percentile endpoints, paired differences and valid counts |
| Behavioral slices | `results/slices/final_test_behavioral_slices.csv` | slice/group/model, support, metrics and threshold outcomes |
| Feature overlap diagnostic | `results/diagnostics/duplicate_profile_overlap.json` | Final Test overlap count and fraction |
| CV evidence and selected configurations | `results/cv/cv_fold_results.csv`, `results/cv/selected_models.json` | fold AP/status/configuration and selected configuration |

The verifier also checked the frozen protocol and split hashes against `results/protocol_v1.0_freeze.json`.

## Methods and tolerances

The independent script is `scripts/verify_phase2_independently.py`. It reads saved row-level predictions and the original source workbook, checks frozen row order, unique IDs, source labels, split labels, probability validity, and the expected 4,500 rows (995 positives, 3,505 negatives). It uses scikit-learn for AP, ROC-AUC, and Brier; ECE is separately implemented with ten equal-width bins, left-closed/right-open intervals, with score 1 included in the final bin and empty bins contributing zero.

Threshold flags use the exact locked rule `score >= threshold`, with FP cost 1 and FN cost 5 for the primary scenario. Sensitivity scenarios use each saved threshold and FN:FP ratio. No Test-driven threshold selection was performed. Bootstrap metrics were recalculated on the exact preserved index matrix, and paired differences were formed replicate by replicate using common draws.

Numerical comparisons use absolute tolerance `1e-12`, appropriate because the saved probability CSV retains substantially more precision than needed to reproduce these metrics. All 411 saved-vs-recomputed floating checks passed this tolerance. Full precision results and comparisons are in `results/verification/recomputed_test_metrics.csv`, `reported_metric_comparison.csv`, `bootstrap_interval_comparison.csv`, and related CSVs.

## Final Test metrics

| Model | Probability | AP | ROC-AUC | Brier | ECE |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Raw | 0.55740871857107654 | 0.76938386655101465 | 0.13527650760502455 | 0.013701378659334263 |
| Logistic Regression | Sigmoid | 0.55740871857107654 | 0.76938386655101465 | 0.13553424200545727 | 0.015377813207915522 |
| Random Forest | Raw | 0.57499965508284046 | 0.78200632262596859 | 0.13274187727259637 | 0.013152649925227165 |
| Random Forest | Sigmoid | 0.57499965508284046 | 0.78200632262596859 | 0.13367732672263394 | 0.017845529091173856 |
| HistGradientBoosting | Raw | 0.57984211138337038 | 0.78815030932121388 | 0.13199677274145449 | 0.011130329866857577 |
| HistGradientBoosting | Sigmoid | 0.57984211138337038 | 0.78815030932121388 | 0.13191858436139281 | 0.010602799208784168 |

All six rows reproduce the saved Phase 2 metrics within `1e-12` (differences are at floating-point roundoff scale). Calibration worsened Brier and ECE for Logistic Regression and Random Forest, and improved both slightly for HistGradientBoosting; calibration was not universally beneficial. HGB has the highest AP and ROC-AUC. Under the locked primary thresholds, Random Forest had the lowest test loss: 3,528 versus HGB's 3,541, a point difference of 13 loss units. This point comparison alone does not establish statistical superiority.

## Locked threshold and sensitivity verification

| Model | Locked threshold | TP | FP | TN | FN | Precision | Recall | FPR | Flagged | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.56718613443475607 | 298 | 118 | 3,387 | 697 | 0.71634615384615385 | 0.29949748743718591 | 0.033666191155492152 | 416 (9.2444%) | 3,603 |
| Random Forest | 0.57138892192274748 | 312 | 113 | 3,392 | 683 | 0.73411764705882354 | 0.31356783919597991 | 0.032239657631954348 | 425 (9.4444%) | 3,528 |
| HistGradientBoosting | 0.59152697416395350 | 309 | 111 | 3,394 | 686 | 0.73571428571428577 | 0.31055276381909547 | 0.03166904422253923 | 420 (9.3333%) | 3,541 |

All primary threshold values, confusion counts, derived rates, and losses reproduce. All 27 saved sensitivity thresholds and their test outcomes also reproduce.

## Bootstrap verification

The saved archive contains 2,000 draws of 4,500 indices, seed 2026, and the exact frozen Test row order. The saved index matrix exactly matches `numpy.random.default_rng(2026).integers(..., dtype=int16)`. All six raw/calibrated score vectors were recomputed from the same draw in each replicate. Every saved replicate metric, all reported model intervals, and all paired-difference point estimates and percentile endpoints matched within `1e-12`. There were zero one-class draws; all 2,000 replicates were valid for AP and ROC-AUC. This is an exact reproduction of the preserved draws, not a fresh bootstrap approximation.

## Critical slices and discrepancy

The independently reconstructed all-six-PAY-nonpositive slice has 3,042 rows and 357 positives. Each primary locked threshold yields zero true positives (LR/RF/HGB = 0/0/0), matching the saved slice output and the corrected Phase 2 narrative.

The `PAY_0 == 0` slice has 2,229 rows and 285 positives. Verified TP counts are LR/RF/HGB = 0/1/0. The original Phase 2 narrative incorrectly stated all three models flagged zero records and all 285 positives were false negatives. Saved predictions, the locked threshold and saved slice CSV agree that RF has TP=1 (FN=284, FP=0); only the narrative was wrong.

The student manually inspected this RF observation in the source workbook, frozen split, saved prediction CSV and locked threshold record: original zero-based row index 20639, source ID 20640, frozen Final Test position 87, target 1, six PAY values `[0, 0, 3, 2, 2, 2]`, calibrated probability approximately 0.7075378319340548 and threshold approximately 0.5713889219227475. This one observation is in the `PAY_0 == 0` slice but outside the all-six-PAY-non-positive slice because several other PAY states are positive. The manual check is limited to this record; it is not a manual reanalysis of all predictions or statistical metrics.

The two PAY-based subgroups overlap but are not equivalent. The all-six-PAY-non-positive slice retains TP=0/0/0 among its 357 positives. The `PAY_0 == 0` slice includes additional observations whose other PAY states are positive. Zero flags under a threshold describes review coverage, not inability to rank risk; the Random Forest TP also shows why the original universal-zero statement was too broad.

### Phase 3A-1b correction audit

- Original incorrect statement: all three models flagged zero observations in `PAY_0 == 0`, making all 285 positives false negatives.
- Corrected values: LR/RF/HGB TP = 0/1/0. The separate all-six-PAY-non-positive group remains 0/0/0 among 357 positives.
- Corrected file: `docs/phase2_results.md`, the two PAY-slice bullets and subgroup relationship paragraph.
- Notebook Markdown: not affected; inspection found no corresponding narrative claim.
- README, figure captions, and `scripts/finalize_reports.py`: no incorrect claim found, so no changes required. The report generator computes slice outputs from saved records and does not generate the incorrect sentence.
- Verifier: its slice oracle now explicitly checks full expected TP vectors (0/1/0 and 0/0/0), in addition to independently recalculating slice membership and threshold outcomes and comparing them with saved slice results.
- Verification progression: initial run returned FAIL due to the narrative inconsistency; after the Phase 3A-1b correction, the verifier and tests were rerun and passed. Actual commands and outcomes are recorded in the Automated checks and execution section below.

## Other checks

- Exact 19-feature overlap with Train+CV: 127 of 4,500 Test rows (2.8222%), including 57 positive labels; agrees with the saved overlap diagnostic. Matching feature profiles do not establish repeated identities.
- Recomputed CV mean-AP aggregation reproduces the saved selected configuration for all three models, using the saved fold results only.
- Protocol wording erratum: **confirmed**. Section 8 reverses the signs. Under delta = compared model minus Logistic Regression, positive AP/ROC-AUC and negative Brier/ECE differences favor the compared model. See `docs/evaluation_protocol_v1.0_erratum.md`. The correction is explanatory and does not alter code or numerical evidence.
- Frozen protocol SHA-256: `03ffa36a02fcdbbf8df8484ffdb84f58e063819899409812cb4e46a87abae418`; matches the freeze manifest. Frozen split SHA-256 also matches.

## Automated checks and execution

Environment: `nussem1`, Python 3.10.18, NumPy 2.2.6, pandas 2.3.1, scikit-learn 1.7.1. The script does not import `scripts/experiment.py` or its calculation helpers.

Commands executed. For portability, the local Conda installation prefix used during execution is represented with `$(conda info --base)`; command sequence and reported outcomes are unchanged:

```text
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate nussem1 && python scripts/verify_phase2_independently.py
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate nussem1 && python -m pytest -q tests/test_independent_verification.py tests/test_experiment.py
git diff --check
shasum -a 256 docs/evaluation_protocol_v1.0.md
```

Historical progression: the initial Phase 3A-1 verification returned FAIL because of the narrative inconsistency. A first implementation run also exposed and corrected a slice count comparison bug before that initial result was finalized. In Phase 3A-1b, the Phase 2 narrative and verifier oracle were corrected; the verifier then completed with exit status 0 and verdict **PASS**. It reported 52 critical-slice checks, 411 floating comparisons, zero failed slice, bootstrap, metric, or CV checks, and exact match to all preserved bootstrap draws and intervals. The combined test command returned **20 passed in 23.40s**. `git diff --check` passed. The protocol SHA-256 was `03ffa36a02fcdbbf8df8484ffdb84f58e063819899409812cb4e46a87abae418`, matching the freeze manifest.

## Verdicts and limitations

| Area | Verdict |
|---|---|
| Row/ID/label alignment and score validity | PASS |
| Six Final Test metric rows | PASS |
| Primary locked thresholds and 27 sensitivity scenarios | PASS |
| Exact paired bootstrap values and intervals | PASS |
| Critical slice counts and corrected narrative | PASS: TP vectors are 0/1/0 and 0/0/0 for the respective distinct slices |
| 19-feature profile overlap | PASS |
| CV selection aggregation | PASS |
| Frozen protocol wording | ERRATUM REQUIRED; original protocol hash preserved |
| Overall after Phase 3A-1b | **PASS** |

This verification checks saved artifacts and arithmetic, not source-data provenance beyond the included workbook, model implementation correctness beyond saved CV summaries, historical or current population validity, causality, operational fairness, or deployment readiness. The student manually verified the single RF observation as documented above; this limited review does not imply manual review of all records or independent recalculation of the statistical metrics. No model retraining occurred.
