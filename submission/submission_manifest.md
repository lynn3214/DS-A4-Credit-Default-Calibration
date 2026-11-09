# DS-A4 Submission Preparation Manifest

**Phase:** 3B candidate package approved by the student. No upload, platform submission, commit, tag, or push has been performed. The student will personally handle Google Drive and SPAIQ actions.

## Requirement sources and confidence

- The official assignment catalog requires a top-to-bottom executable notebook or equivalent code, exported PDF, reproduction commands, data/source/license/split/seed/dependency/runtime information, evidence-supported analysis, failure or counterexample discussion, and an accurate AI-use record: <https://tjluo-ucas.github.io/ds/en/assignments/>.
- The course submission guide says written materials and supporting evidence belong in Google Drive; runnable code and documentation belong on GitHub; and the course platform record includes the student's full name and student ID, submission type/title/assignment number, a brief summary, and Drive/GitHub links. It also specifies Viewer access for `tiejianluo@gmail.com` to the Drive course folder and instructor access to GitHub: <https://tjluo-ucas.github.io/ds/en/submission-guide/>.
- The syllabus reinforces the notebook/PDF, reproducibility, uncertainty, counterexample, and AI-use expectations: <https://tjluo-ucas.github.io/ds/en/syllabus/>.
- `DS_A4_TASK_SPEC.md` records a 150–300-word summary requirement from the previously supplied instructor assignment. The original instructor handout itself is not present in this repository, and the public catalog does not state a word range. The approved summary uses 150–300 words as a provisional target, not as a separately verified public requirement.
- The current matching SPAIQ assignment page could not be inspected through the available public web access. No platform-specific fields beyond the public course guide are asserted here; the student should check the live task before submission.

## Candidate deliverables and readiness

| Deliverable | Status | Evidence / remaining action |
|---|---|---|
| Current approved Notebook | PASS | `notebooks/DS_A4_model_comparison_calibration.ipynb` has the latest reviewed scientific narrative and saved outputs. Only narrative-status wording changed in Phase 3B; all code cells and saved numerical outputs are checked for preservation. |
| Candidate PDF | PASS | `submission/DS_A4_model_comparison_calibration_final_candidate.pdf`, 8 pages, 885,590 bytes; SHA-256 `0b91377963cb26156ce2a9e9baa07190daa50c8bf93fd5a216aa26d5a5c80015`. Exported from a temporary Notebook copy without code input or re-execution. The shared SVG was embedded for export. All pages were rendered and visually inspected; charts, tables, axes, and diagram appear, with no clipping, broken images, blank pages, or stale pre-correction PAY narrative. The student reviewed and approved it. Five embedded output images lack alternative text in the source outputs. |
| SPAIQ results summary | PASS | `submission/spaiq_results_summary.md`, 221 words. Values are from saved verified results; it contains no invented identity or submission link. The student reviewed and approved it. |
| Reproduction instructions | PASS with disclosed data retrieval step | README and `data/raw/README.md` specify setup, official archive URL, expected byte size/hash, ZIP integrity check, frozen seeds, run modes, and verification commands. The raw UCI archive is intentionally not tracked; a reviewer must download it to the documented path before audit, experiment, or verifier commands. |
| Submission manifest | PASS | This file maps intended GitHub, Drive, and SPAIQ contents and identifies unconfirmed actions/requirements. The student reviewed and approved the manifest; status wording is updated to reflect that review. |
| GitHub publication | REVISE | The configured `origin` is `https://github.com/lynn3214/DS-A4-Credit-Default-Calibration.git`. The current worktree contains uncommitted Phase 3A/3B changes, so a stable URL for this candidate state does not yet exist. The student should review the staged changes, create the approved commit, and push the current branch before using its stable repository/commit URL. No commit, tag, or push was made in this phase. The original raw ZIP remains excluded by `.gitignore`. |
| Google Drive | BLOCKED pending student action | The student will upload the approved PDF, Markdown reports, diagram, and essential supporting evidence; organize them by assignment; share the folder with `tiejianluo@gmail.com` as Viewer; and verify access. Nothing was uploaded and permissions were not checked here. |
| SPAIQ record | BLOCKED pending student/platform access | The student will check the live DS-A4 task, enter their own identifying information and assignment details, and provide the approved summary plus Drive and stable GitHub links. No record was submitted here. |

**Overall readiness: READY for local Git commit preparation; REVISE for course delivery.** The student has reviewed and approved the candidate PDF, summary, and manifest. Google Drive upload/access checks, final repository publication, and SPAIQ actions remain student-managed and incomplete; live platform-specific requirements remain unverified.

## Inspection of the pre-existing PDF

`notebooks/DS_A4_model_comparison_calibration.pdf` existed before this Phase 3B export and was preserved byte-for-byte. It is 829,976 bytes (current SHA-256 `99713f17b6dfd068a31d3c2eb9b1b64bfb6d88524ded1cebca67b13997674534`) and its text is extractable. The system `file` utility reports 8 pages, while Ghostscript rendered 20 pages; that page-count discrepancy makes its pagination unreliable. Its scientific narrative includes the corrected PAY-slice counts and the Phase 3A-3 content, but it still says the revised narrative awaits review. Visual inspection also found a broken data-flow SVG image, visible code and verbose JSON outputs, and wide tables split or clipped across pages. It is not the final candidate and should not be published.

## A. Intended GitHub contents

After the student approves the package and publishes the reviewed working tree, include:

- `README.md`, `requirements.txt`, `DS_A4_TASK_SPEC.md`, and `data/raw/README.md`;
- `notebooks/DS_A4_model_comparison_calibration.ipynb` and the approved final PDF from `submission/` (do not include the stale pre-existing `notebooks/DS_A4_model_comparison_calibration.pdf`);
- `scripts/`, `tests/`, frozen protocol, protocol erratum, dataset audit, Phase 2 results, independent verification report, AI-use log, and frozen data-flow diagram;
- saved base models and sigmoid calibrators plus their metadata, predictions, metrics, CV summaries, bootstrap evidence, decisions, slices, verification outputs, and figures under `results/` that support reproducibility and review;
- `submission/spaiq_results_summary.md` and this manifest as approved preparation documentation.

Keep `data/raw/uci_default_credit_card_350.zip` excluded as previously instructed and as enforced by `.gitignore`. Also exclude local caches, temporary PDF-rendering images, temporary export notebooks, Chromium downloads, and unrelated local files. The UCI source is licensed CC BY 4.0 and must be retrieved using the source/hash instructions in `data/raw/README.md`.

## B. Intended Google Drive contents

Upload only after approval:

- the approved final Notebook PDF;
- `docs/phase2_results.md`, `docs/phase3a_independent_verification.md`, `docs/evaluation_protocol_v1.0.md`, and `docs/evaluation_protocol_v1.0_erratum.md`;
- `docs/frozen_data_flow_v1.0.svg` and the essential compact figures/tables that support the reported conclusions;
- `docs/ai_use_log.md`, `submission/spaiq_results_summary.md`, and any other evidence specifically requested by the live task.

Do not upload the raw data archive, caches, temporary exports, duplicate PDFs, or unnecessary intermediate files. Create an assignment-organized folder and apply the course guide's Viewer-sharing requirement; the folder and access have not been created or verified in this phase.

## C. SPAIQ information for the student to enter

- Student full name and student ID (student-supplied; not recorded in this manifest).
- Submission type/title and DS-A4 assignment number/week as shown on the live task.
- The approved summary (currently 221 words).
- The Google Drive folder link after upload and access verification.
- The GitHub repository and stable approved commit/tag link after publication.
- Any additional fields shown by the current DS-A4 task page; that page was not accessible for independent confirmation here.

No student identifiers, Drive links, upload confirmations, permissions, or platform submission status have been invented. The configured GitHub origin is listed above.

## Reproducibility audit notes

- `requirements.txt` pins tested direct dependencies and now includes `ipykernel`, `nbconvert`, and `jupyter_core` needed by the documented notebook execution path. The measured research runtime is Python 3.10.18; dependency versions are also recorded in `results/phase2_execution_report.json`.
- The original UCI archive is read directly from the ZIP at `data/raw/uci_default_credit_card_350.zip`. `data/raw/README.md` gives an official `curl` command, expected 5,539,494-byte size, SHA-256, and `unzip -t` check. Retrieval remains a manual prerequisite; no downloader script exists.
- `scripts/audit_dataset.py` generates a provisional audit/split, while the frozen split/folds actually used by Phase 2 are saved in `results/audit/proposed_splits.json` and protected by `results/protocol_v1.0_freeze.json`. Do not rerun the audit to replace frozen Phase 2 indices.
- The full `scripts.experiment` path performs 180 CV fits and writes experimental outputs. The default notebook saved-results mode calls the finalizer and can refresh derived outputs; README now directs reviewers to use an isolated project copy for notebook execution. No clean full-training reproduction was run in Phase 3B.
- PDF export used an export-only temporary Notebook copy with `RUN_FULL_EXPERIMENT = False`, no code inputs displayed, and selected saved outputs reformatted for page width. No source Notebook code was executed, and no model or statistical analysis was rerun to produce the PDF.

## Phase 3B readiness report

| Area | Status | Basis |
|---|---|---|
| Scientific evidence integrity | PASS | Protocol, split, models, calibrators, saved predictions, bootstrap evidence, thresholds, and original result files were preserved. |
| PDF technical and visual review | PASS | Candidate is readable, 8 rendered pages; diagram and relevant plots/tables appear without visible rendering failures. The student approved the candidate. |
| Summary consistency | PASS | Summary agrees with saved Final Test metric and loss records and states the 13-unit gap as a point comparison without a direct paired interval. The student approved the summary. |
| Reproduction documentation | PASS with disclosed data prerequisite | Environment, source/license/hash, seeds, run modes, and checks are documented; reviewer must retrieve the excluded archive. |
| Live SPAIQ requirements | BLOCKED / not independently confirmed | The public pages describe the general submission record; the current matching task and any additional fields could not be inspected. |
| External delivery | REVISE / student-managed | The student approved the local candidate package. No Drive/GitHub publication or SPAIQ submission occurred in this phase. |

No external submission action was performed in this phase. The student will personally complete the Drive and SPAIQ steps after checking the live task requirements.
