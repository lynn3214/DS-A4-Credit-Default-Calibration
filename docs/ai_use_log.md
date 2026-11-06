# AI use log — Phases 1 and 2

## Assistance and generated artifacts

- Tool: Codex (OpenAI coding assistant), 2026-11-06. Read the supplied task and `DS_A4_TASK_SPEC.md`; checked the official UCI dataset page; downloaded its official archive; bootstrapped the repository; wrote the audit script and project documentation.
- AI-generated or modified: `.gitignore`, `requirements.txt`, README, raw-data instructions, `scripts/audit_dataset.py`, `docs/dataset_audit.md`, `docs/evaluation_protocol_v0.1.md`, this log, and acquisition evidence. The supplied specification was preserved.
- Recommendations accepted: preserve the official archive unchanged, exclude it from Git, keep demographic predictors outside the primary allowlist, preserve unexplained codes pending review, and keep the protocol provisional.
- One initial target-header assumption was wrong: the workbook uses `default payment next month`; the script was corrected to use the actual source header.

## Verification actually performed

- Read the full project specification.
- Checked the UCI page for dataset identity, source-reported row/feature counts, variable descriptions, missing-value claim, and CC BY 4.0 license.
- Downloaded the official ZIP archive; inspected its member list; verified the ZIP CRC with `unzip -t`; computed archive size and SHA-256.
- Activated `nussem1`; installed missing `xlrd==2.0.1` after network installation authorization. Runtime versions: Python 3.10.18, pandas 2.3.1, NumPy 2.2.6, scikit-learn 1.7.1, xlrd 2.0.1.
- Ran `python scripts/audit_dataset.py` successfully after correcting the target header. It inspected schema, labels, missing/nonfinite values, numeric ranges, full category frequencies, IDs, exact rows, duplicate predictors and label conflicts, feature allowlist, split coverage/support, major audit slices, and five common stratified CV folds. It does not fit or train any models.
- Ran the audit twice. The generated split-index JSON SHA-256 matched on both runs: `ded744149bd5afead62f3355d773eded3e9be1c8a53a5deabee38e104f42b1ed`.

## Human review not yet completed

The student had not yet independently verified the downloaded source, code, findings, or research-design choices at the end of Phase 1. Phase 2 decisions were authorized in the supplied instruction. Independent review of the experiment artifacts is still pending.

## Phase 2 assistance and generated artifacts

- Tool: Codex (OpenAI coding assistant), 2026-11-06. Read the complete Phase 2 request and required Phase 1 artifacts; froze Protocol v1.0 before model fitting; implemented the model/calibration/decision pipeline, reporting finalizer, automated checks, notebook, results documentation, figures, and this log update.
- AI-generated or modified: `docs/evaluation_protocol_v1.0.md`, `results/protocol_v1.0_freeze.json`, `scripts/experiment.py`, `scripts/finalize_reports.py`, `tests/test_experiment.py`, `notebooks/DS_A4_model_comparison_calibration.ipynb`, `requirements.txt`, README, `docs/phase2_results.md`, phase 2 result tables/figures/models/predictions, and `results/logs/phase2_execution_incidents.md`.
- Decisions followed from the student's Phase 2 instruction: preserve the existing row-level split/folds and duplicates; use only the 19-feature allowlist; retain PAY states as separate one-hot categories; assume the end-of-September cutoff; use the fixed three model grids, independent sigmoid calibration, paired bootstrap, cost/capacity rules, and supplementary demographic error counts.

## Phase 2 verification actually performed

- Verified protocol, split, and source SHA-256 values before training and again during report finalization.
- Verified 180/180 CV fold/configuration records: 60 per model, all status `ok`; no convergence warnings. Selected model artifacts report unchanged base-model predictions after fitting calibration.
- Verified row-aligned final-test indices, IDs, and labels against the frozen split. Verified 2,000 shared bootstrap draws, seed 2026, replicate count, and one-class-resample count.
- Ran `MPLCONFIGDIR=/tmp/ds_a4_mpl python -m scripts.finalize_reports` successfully. It used existing fitted models/predictions and performed no model training.
- Ran `MPLCONFIGDIR=/tmp/ds_a4_mpl python -m pytest -q tests/test_experiment.py`: **10 passed**, including checks that saved bootstrap metrics recompute from the shared stored test-row draws and that locked thresholds were applied unchanged.
- Executed the final notebook with `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=-1 --inplace ...` in `nussem1`. The clean kernel defaulted to `RUN_FULL_EXPERIMENT=False`, validated/finalized saved results, and rendered tables/figures without refitting.
- The notebook's initial clean-kernel attempt exposed a missing manifest field; the report finalizer was corrected to record one-class bootstrap resamples. An initial exact floating-point ECE test was changed to a tolerance assertion. Two artifact-level checks were added for paired bootstrap resampling and locked test thresholds; the final targeted suite passed (10 tests).

## Execution incidents and limits

- Jupyter kernel startup in the sandbox failed while binding a local port (`Operation not permitted`); the clean-kernel execution required escalation.
- A repeated full-search attempt was interrupted during Random Forest CV after duplicate notebook execution attempts. The complete CV aggregate was validated at 180 successful fits; the interrupted retry was not used as final evidence. Details are in `results/logs/phase2_execution_incidents.md`.
- A per-point curve plotting implementation was replaced with grouped plotting to reduce rendering overhead. The experiment method, protocol, model choices, thresholds, and test outcomes were not changed by this reporting fix.
- The clean-kernel notebook verification replayed verified saved outputs; it did not repeat the 1,155 seconds of cumulative CV fit time. `RUN_FULL_EXPERIMENT=True` enables an exact full rerun from the notebook or `python -m scripts.experiment`.

## Independent human review still required

The student has not independently verified the code, model artifacts, findings, or written interpretation in this log. Review the frozen protocol hash, all 180 fold rows, predictions, paired intervals, threshold records, sparse slice results, and the documented execution incidents before Phase 3. No PDF or submission was created. A repository commit was later made at the student's request; no tag was created.
