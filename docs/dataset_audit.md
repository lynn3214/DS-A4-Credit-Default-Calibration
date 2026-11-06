# Dataset acquisition and audit

**Audit executed successfully in the `nussem1` Conda environment on 2026-11-06.** The machine-readable results are in `results/audit/audit.json`; exact proposed row indices and CV folds are in `results/audit/proposed_splits.json`. The protocol remains provisional.

## Provenance

- Dataset: UCI Machine Learning Repository, *Default of Credit Card Clients*, ID 350; DOI [10.24432/C55S3H](https://doi.org/10.24432/C55S3H).
- Official page: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Download URL: `https://archive.ics.uci.edu/static/public/350/default%2Bof%2Bcredit%2Bcard%2Bclients.zip`
- Retrieved: 2026-11-06 (Asia/Singapore) from the official UCI host.
- Archive: `data/raw/uci_default_credit_card_350.zip`, 5,539,494 bytes; SHA-256 `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`.
- ZIP contains one original workbook, `default of credit card clients.xls`, 5,539,328 uncompressed bytes. ZIP integrity check passed. Audit reads the workbook directly from the ZIP and does not rewrite it.
- License: CC BY 4.0. Attribute Yeh (2009), dataset title and UCI; indicate adaptations if any are made. None were made in this phase.

## Schema and target

The workbook has **30,000 rows and 25 columns**: ID, 23 explanatory fields, and the target. UCI describes 30,000 instances and 23 features; the additional physical column is ID. All 25 columns loaded as `int64`. Exact expected column sequence passed. `PAY_0` is present and `PAY_1` absent. The target's actual workbook header is `default payment next month` (spaces), rather than the dot-separated header anticipated in the initial plan. The audit uses the literal source header; this naming discrepancy should be reflected in any later output.

| Target value | Count | Share |
| --- | ---: | ---: |
| 0 | 23,364 | 77.88% |
| 1 | 6,636 | 22.12% |

No missing values or non-finite numeric values were found in any column. Target values are binary and valid. These findings were measured from the workbook, not inferred from UCI's page.

## Complete observed categorical frequencies

Frequencies below are `code: row count`; absent codes have zero frequency. These values are preserved, not recoded.

| Field | Observed frequencies |
| --- | --- |
| SEX | 1: 11,888; 2: 18,112 |
| EDUCATION | 0: 14; 1: 10,585; 2: 14,030; 3: 4,917; 4: 123; 5: 280; 6: 51 |
| MARRIAGE | 0: 54; 1: 13,659; 2: 15,964; 3: 323 |
| PAY_0 | -2: 2,759; -1: 5,686; 0: 14,737; 1: 3,688; 2: 2,667; 3: 322; 4: 76; 5: 26; 6: 11; 7: 9; 8: 19 |
| PAY_2 | -2: 3,782; -1: 6,050; 0: 15,730; 1: 28; 2: 3,927; 3: 326; 4: 99; 5: 25; 6: 12; 7: 20; 8: 1 |
| PAY_3 | -2: 4,085; -1: 5,938; 0: 15,764; 1: 4; 2: 3,819; 3: 240; 4: 76; 5: 21; 6: 23; 7: 27; 8: 3 |
| PAY_4 | -2: 4,348; -1: 5,687; 0: 16,455; 1: 2; 2: 3,159; 3: 180; 4: 69; 5: 35; 6: 5; 7: 58; 8: 2 |
| PAY_5 | -2: 4,546; -1: 5,539; 0: 16,947; 1: 0; 2: 2,626; 3: 178; 4: 84; 5: 17; 6: 4; 7: 58; 8: 1 |
| PAY_6 | -2: 4,895; -1: 5,740; 0: 16,286; 1: 0; 2: 2,766; 3: 184; 4: 49; 5: 13; 6: 19; 7: 46; 8: 2 |

UCI's text describes education 1–4, marriage 1–3, and repayment codes -1 and 1–9. It does not explain all observed values. `EDUCATION` contains 0/5/6 (345 records), `MARRIAGE` contains 0 (54), and `PAY_*` contains -2 and 0. No meaning has been invented for these values. Review and approve a consistent model representation before training. Some tail `PAY_0` categories have only one to a few records in held-out subsets, making subgroup metrics unstable.

Numeric ranges are saved per field in `audit.json`. Highlights: `LIMIT_BAL` 10,000–1,000,000; `AGE` 21–79; all repayment fields -2–8; bill amounts include negatives (across fields, minima range from -339,603 to -69,777); payment amounts are nonnegative (per-field maxima 426,529–1,684,259). Negative bill statements are retained; interpretation should be reviewed rather than treated as an automatic data error.

## Identity and duplicate audit

No IDs are missing or duplicated. There are no exact duplicate full rows. However, distinct IDs can have identical predictors:

| Feature definition | Duplicate-vector groups | Rows in those groups | Groups with conflicting target labels |
| --- | ---: | ---: | ---: |
| All 23 predictors (exclude ID and target) | 52 | 108 | 21 |
| Primary 19-feature allowlist | 134 | 951 | 85 |

No records were removed. Conflicting labels among identical predictors are plausible observational ambiguity, not proof of bad data. Random row splitting can place identical feature vectors in both train and held-out partitions, which may make evaluation less independent. Before freezing the protocol, decide whether to retain the proposed row-level stratification and disclose this limitation, or group identical full predictor vectors in a suitably designed split. Grouping may complicate exact stratification and is a protocol change requiring revalidation.

## Leakage and feature policy

The tested primary allowlist is exactly `LIMIT_BAL`, `PAY_0`, `PAY_2`–`PAY_6`, `BILL_AMT1`–`BILL_AMT6`, and `PAY_AMT1`–`PAY_AMT6`. Target and ID are excluded. `SEX`, `EDUCATION`, `MARRIAGE`, and `AGE` are reserved for cautiously aggregated supplementary error review, not primary prediction. No target-derived transformation was introduced.

The assumed end-of-September 2005 decision cutoff is not verified operational timing: the UCI documentation supplies monthly variable descriptions, but no availability timestamps. Confirm the intended simulated prediction moment and feature availability. This historical random split cannot establish temporal generalization or current-borrower performance. The target is next-month default; data do not show that intervention prevents default.

## Proposed split feasibility (not frozen)

Seed 42 stratified split allocations were generated; subsets are disjoint and cover all 30,000 zero-based row positions. Each partition has both classes. Five common shuffled stratified folds were generated within Train+CV, and every train and validation fold contains both classes. Folds are feasible by count, although some rare category slices remain very small.

| Partition | Rows | Negative | Positive | Positive share |
| --- | ---: | ---: | ---: | ---: |
| Train+CV | 18,000 | 14,018 | 3,982 | 22.12% |
| Calibration | 4,500 | 3,505 | 995 | 22.11% |
| Threshold Selection | 3,000 | 2,336 | 664 | 22.13% |
| Final Test | 4,500 | 3,505 | 995 | 22.11% |

Major slice support is recorded per partition in `audit.json`, including PAY_0, any positive repayment delinquency state, and train-derived LIMIT_BAL tertiles. The smallest PAY_0 cells in calibration/threshold/test are 1–6 records. Do not interpret unstable small-cell performance. `proposed_splits.json` contains full row positions and IDs; fold positions are local to the listed Train+CV order. Split indices remain provisional and must not be treated as a frozen protocol.

## Reproduction and environment

From the project root, in the active `nussem1` environment:

```sh
python scripts/audit_dataset.py
```

Verified runtime: Python 3.10.18, pandas 2.3.1, NumPy 2.2.6, scikit-learn 1.7.1, xlrd 2.0.1; macOS 15.5 arm64. The audit ran successfully twice. The generated split JSON SHA-256 was identical both times: `ded744149bd5afead62f3355d773eded3e9be1c8a53a5deabee38e104f42b1ed`. Execution time is recorded in the JSON and varies slightly. The source archive checksum was unchanged.

## Gate

**REVISE.** Data acquisition, deterministic integrity checks, and split feasibility succeeded. Before freezing, resolve ambiguous codes, decide how to handle identical predictor vectors in separate partitions, confirm the prediction cutoff/field availability, and review the proposed error-cost, capacity, model-budget, and calibration details. No model training, calibration, threshold optimization, or test performance evaluation has occurred.
