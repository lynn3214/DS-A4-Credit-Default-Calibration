# DS-A4: Model Comparison and Calibration Lab

This project studies next-month credit-card default risk in the UCI Default of Credit Card Clients dataset (ID 350; DOI [10.24432/C55S3H](https://doi.org/10.24432/C55S3H)). The Phase 2 comparison evaluates Logistic Regression, Random Forest, and HistGradientBoosting on frozen paired folds, then examines independent sigmoid calibration and hypothetical cost/capacity rules. It is an offline predictive risk-review simulation; it does not show that intervention prevents default or justify operational lending use.

## Status

Phase 1 acquisition and audit are complete. Evaluation Protocol v1.0 was frozen before model fitting. The three-model CV searches, refits, calibration, threshold decisions, Final Test metrics, paired bootstrap, slice analysis, and figures are present under `results/`. The runnable notebook has been executed from a clean `nussem1` kernel to validate and render the saved experiment outputs. Independent student/GPT review is still pending. The final PDF and submission materials have not been produced. The original data archive is intentionally not tracked by Git.

Key observed results: HistGradientBoosting had the highest Final Test AP (0.5798) and ROC-AUC (0.7882). Its sigmoid probabilities had Brier 0.1319 and ECE 0.0106. Sigmoid calibration did not improve every model's Brier/ECE. Under the hypothetical 1:5 FN:FP and 10% selection-set capacity, test flagged fractions were 9.2–9.4%; Random Forest had the lowest test surrogate loss of the three. See [Phase 2 results](docs/phase2_results.md) for intervals, failure slices, and limitations.

## Layout

- `DS_A4_TASK_SPEC.md`: original assignment and Phase 1 specification, preserved.
- `data/raw/`: original UCI ZIP, excluded from Git; provenance and SHA-256 are recorded in the audit evidence.
- `scripts/audit_dataset.py`: Phase 1 source-data audit and provisional split generator.
- `scripts/experiment.py`: frozen CV, refit, calibration, threshold, test, and bootstrap pipeline.
- `scripts/finalize_reports.py`: validates completed experiment artifacts and derives slices/figures without refitting models.
- `docs/evaluation_protocol_v1.0.md`: frozen Phase 2 evaluation protocol.
- `docs/dataset_audit.md`, `docs/phase2_results.md`, `docs/ai_use_log.md`: source audit, experiment findings, and assistance/verification record.
- `notebooks/DS_A4_model_comparison_calibration.ipynb`: English research notebook. By default it validates and presents existing results without retraining; set `RUN_FULL_EXPERIMENT = True` in its first code cell to repeat all 180 frozen CV fits.
- `results/audit/`: Phase 1 machine-readable evidence and frozen split indices.
- `results/cv/`, `results/models/`, `results/predictions/`, `results/statistics/`, `results/calibration/`, `results/slices/`, `results/diagnostics/`, `results/decisions/`, `results/figures/`: Phase 2 configurations, fitted artifacts, aligned outputs, statistics, tables, and figures.

## Reproducibility in `nussem1`

```sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nussem1
python -m pip install -r requirements.txt

# Full frozen model run (180 CV fits; one worker; runtime depends on hardware).
MPLCONFIGDIR=/tmp/ds_a4_mpl python -m scripts.experiment

# Validate finalized reports/figures from completed model outputs without refitting.
MPLCONFIGDIR=/tmp/ds_a4_mpl python -m scripts.finalize_reports

# Targeted automated checks.
MPLCONFIGDIR=/tmp/ds_a4_mpl python -m pytest -q tests/test_experiment.py
```

To execute the notebook cleanly and render the saved outputs, use:

```sh
MPLCONFIGDIR=/tmp/ds_a4_mpl IPYTHONDIR=/tmp/ds_a4_ipython jupyter nbconvert \
  --to notebook --execute --ExecutePreprocessor.enabled=True \
  --ExecutePreprocessor.timeout=-1 --inplace \
  notebooks/DS_A4_model_comparison_calibration.ipynb
```

The original source archive SHA-256 is `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`. The protocol freeze and split hashes are recorded in `results/protocol_v1.0_freeze.json`. Tested dependencies are pinned in `requirements.txt`; the measured environment is recorded in `results/phase2_execution_report.json`.

## Dataset and License

This project uses the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository.

**Dataset citation:**

Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C55S3H](https://doi.org/10.24432/C55S3H)

- **Source:** [UCI dataset page](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **License URL:** [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)
- **Dataset ID:** 350

The original UCI archive is preserved unchanged in the local `data/raw/uci_default_credit_card_350.zip` file. It is intentionally excluded from this Git repository and will not be uploaded. All preprocessing, feature selection, experimental splits, and model outputs are produced separately by this project; the original source data is not modified. The dataset is used for academic research and reproducible offline evaluation. The resulting models are not intended for operational credit or lending decisions.
