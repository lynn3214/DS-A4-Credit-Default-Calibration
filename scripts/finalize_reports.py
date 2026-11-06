"""Validate completed frozen experiment outputs and derive reports and figures.

This reporting step never fits or refits a model and never changes a threshold.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from scripts.experiment import (ARCHIVE, BOOTSTRAP_N, BOOTSTRAP_SEED, DEMOGRAPHICS,
                                FEATURES, MODEL_NAMES, PAY, ROOT, TARGET,
                                confusion_metrics, create_figures, load_data,
                                metric_bundle, profile_keys, sha256,
                                verify_frozen_inputs, top_k_flags)


def finalize_reports() -> dict:
    started = time.perf_counter()
    freeze, split_doc = verify_frozen_inputs()
    out = ROOT / "results"
    cv = pd.read_csv(out / "cv/cv_fold_results.csv")
    if len(cv) != 180 or set(cv.status) != {"ok"} or cv.groupby("model").size().to_dict() != {name: 60 for name in MODEL_NAMES}:
        raise RuntimeError("Core CV evidence is incomplete or contains failed configurations")
    selected = json.loads((out / "cv/selected_models.json").read_text())
    decisions = json.loads((out / "decisions/locked_decisions.json").read_text())
    if decisions.get("status") != "locked before final test predictions":
        raise RuntimeError("Threshold decisions are not recorded as locked before test scoring")
    unique_top_k = {}
    for record in decisions.get("top_k", []):
        unique_top_k[(record["model"], float(record["capacity"]))] = record
    decisions["top_k"] = [unique_top_k[(model, capacity)]
                           for model in MODEL_NAMES for capacity in [0.05, 0.10, 0.20]
                           if (model, capacity) in unique_top_k]
    if len(decisions["top_k"]) != len(MODEL_NAMES) * 3:
        raise RuntimeError("Expected one locked Top-K policy for each model/capacity pair")
    (out / "decisions/locked_decisions.json").write_text(json.dumps(decisions, indent=2))
    split = {k: np.asarray(v, dtype=int) for k, v in split_doc["indices_zero_based"].items()}
    frame = load_data()
    train_rows, cal_idx, threshold_idx, test_idx = (split[k] for k in ["train_cv", "calibration", "threshold_selection", "final_test"])
    test_frame = frame.iloc[test_idx]
    test_y = test_frame[TARGET].to_numpy(dtype=int)
    test_predictions = pd.read_csv(out / "predictions/final_test_predictions.csv")
    if not np.array_equal(test_predictions.row_position.to_numpy(dtype=int), test_idx):
        raise RuntimeError("Final-test prediction positions are not aligned to the frozen test indices")
    if not np.array_equal(test_predictions.ID.to_numpy(), test_frame.ID.to_numpy()) or not np.array_equal(test_predictions.y_true.to_numpy(dtype=int), test_y):
        raise RuntimeError("Final-test prediction IDs or labels do not align to frozen records")
    for model in MODEL_NAMES:
        if f"{model}__raw" not in test_predictions or f"{model}__sigmoid" not in test_predictions:
            raise RuntimeError(f"Missing test probabilities for {model}")
        if not json.loads((out / f"models/{model.lower().replace(' ', '_')}_metadata.json").read_text())["base_predictions_unchanged_after_calibration_fit"]:
            raise RuntimeError(f"Calibration separation check failed for {model}")

    train_frame = frame.iloc[train_rows]
    limit_bounds = [float(v) for v in np.quantile(train_frame.LIMIT_BAL, [1/3, 2/3])]
    train_burden = train_frame.BILL_AMT1.to_numpy(dtype=float) / train_frame.LIMIT_BAL.to_numpy(dtype=float)
    burden_bounds = [float(v) for v in np.quantile(train_burden, [1/3, 2/3])]
    burden_valid = bool(np.isfinite(burden_bounds).all() and burden_bounds[0] < burden_bounds[1])
    boundary_doc = {"derived_only_from_train_cv": True, "limit_balance_tertile_cutpoints": limit_bounds,
                    "signed_september_bill_to_limit_ratio": "BILL_AMT1 / LIMIT_BAL; negative values retained",
                    "bill_to_limit_tertile_cutpoints": burden_bounds,
                    "bill_to_limit_slice_status": "included" if burden_valid else "omitted: training tertile cutpoints are not distinct and finite"}
    (out / "slices/train_derived_boundaries.json").write_text(json.dumps(boundary_doc, indent=2))

    pay0 = test_frame.PAY_0.to_numpy()
    pay0_group = np.where(pay0 <= -2, "-2", np.where(pay0 == -1, "-1", np.where(pay0 == 0, "0", ">=1")))
    any_delay = test_frame[PAY].to_numpy().astype(float).max(axis=1) > 0
    limit_tier = np.asarray(pd.cut(test_frame.LIMIT_BAL, [-np.inf, *limit_bounds, np.inf], labels=["low", "middle", "high"], include_lowest=True).astype(str))
    slice_arrays = {"PAY_0 grouped": pay0_group,
                    "Any PAY code > 0": np.where(any_delay, "true", "false"),
                    "LIMIT_BAL train tertile": limit_tier}
    if burden_valid:
        burden = test_frame.BILL_AMT1.to_numpy(dtype=float) / test_frame.LIMIT_BAL.to_numpy(dtype=float)
        slice_arrays["Signed bill-to-limit train tertile"] = pd.cut(
            burden, [-np.inf, *burden_bounds, np.inf], labels=["low", "middle", "high"], include_lowest=True
        ).astype(str)

    slice_rows = []
    for slice_name, labels in slice_arrays.items():
        for group in sorted(np.unique(labels)):
            mask = labels == group
            ys = test_y[mask]
            for model in MODEL_NAMES:
                prob = test_predictions.loc[mask, f"{model}__sigmoid"].to_numpy()
                metrics, _ = metric_bundle(ys, prob)
                threshold = float(decisions["primary"][model]["threshold"])
                errors = confusion_metrics(ys, prob >= threshold, 5.0)
                slice_rows.append({"slice": slice_name, "group": str(group), "model": model, "n": int(mask.sum()),
                                   "positive": int(ys.sum()), "negative": int(len(ys) - ys.sum()), **metrics,
                                   **{f"threshold_{k}": v for k, v in errors.items()}})
    pd.DataFrame(slice_rows).to_csv(out / "slices/final_test_behavioral_slices.csv", index=False)

    age = test_frame.AGE.to_numpy()
    age_band = np.select([age < 30, age < 40, age < 50, age < 60], ["<30", "30-39", "40-49", "50-59"], default="60+")
    demographic_values = {field: test_frame[field].astype(str).to_numpy() for field in ["SEX", "EDUCATION", "MARRIAGE"]}
    demographic_values["AGE_band"] = age_band
    demographic_rows = []
    for field, labels in demographic_values.items():
        for group in sorted(np.unique(labels)):
            mask = labels == group
            ys = test_y[mask]
            for model in MODEL_NAMES:
                p = test_predictions.loc[mask, f"{model}__sigmoid"].to_numpy()
                threshold = float(decisions["primary"][model]["threshold"])
                errors = confusion_metrics(ys, p >= threshold, 5.0)
                demographic_rows.append({"field": field, "group": str(group), "model": model, "n": int(mask.sum()),
                                         "positive": int(ys.sum()), "negative": int(len(ys) - ys.sum()),
                                         "tp": errors["tp"], "fp": errors["fp"], "tn": errors["tn"], "fn": errors["fn"]})
    pd.DataFrame(demographic_rows).to_csv(out / "slices/supplementary_demographic_error_audit.csv", index=False)

    train_profiles = set(profile_keys(frame.iloc[train_rows]))
    overlap_doc = {}
    for part in ["calibration", "threshold_selection", "final_test"]:
        idx = split[part]
        mask = np.asarray([profile in train_profiles for profile in profile_keys(frame.iloc[idx])], dtype=bool)
        overlap_doc[part] = {"n": int(len(mask)), "overlap_n": int(mask.sum()), "overlap_fraction": float(mask.mean()),
                             "nonoverlap_n": int((~mask).sum())}
    (out / "diagnostics/duplicate_profile_overlap.json").write_text(json.dumps(overlap_doc, indent=2))
    test_overlap = np.asarray([profile in train_profiles for profile in profile_keys(test_frame)], dtype=bool)
    overlap_rows = []
    for group_name, mask in [("overlap", test_overlap), ("nonoverlap", ~test_overlap)]:
        ys = test_y[mask]
        for model in MODEL_NAMES:
            prob = test_predictions.loc[mask, f"{model}__sigmoid"].to_numpy()
            metrics, _ = metric_bundle(ys, prob)
            threshold = float(decisions["primary"][model]["threshold"])
            errors = confusion_metrics(ys, prob >= threshold, 5.0)
            overlap_rows.append({"profile_group": group_name, "model": model, "n": int(mask.sum()),
                                 "positive": int(ys.sum()), "negative": int(len(ys) - ys.sum()), **metrics,
                                 **{f"threshold_{k}": v for k, v in errors.items()}})
    pd.DataFrame(overlap_rows).to_csv(out / "diagnostics/test_overlap_group_metrics.csv", index=False)

    sensitivity_rows = []
    for record in decisions["sensitivity"]:
        model = record["model"]
        p = test_predictions[f"{model}__sigmoid"].to_numpy()
        test_errors = confusion_metrics(test_y, p >= float(record["threshold"]), float(record["fn_to_fp_ratio"]))
        sensitivity_rows.append({**record, "test_threshold": record["threshold"],
                                 **{f"test_{k}": v for k, v in test_errors.items()}})
    pd.DataFrame(sensitivity_rows).to_csv(out / "decisions/cost_capacity_sensitivity.csv", index=False)
    top_rows = []
    for record in decisions["top_k"]:
        model = record["model"]
        flags, top_meta = top_k_flags(test_predictions[f"{model}__sigmoid"].to_numpy(), test_idx, float(record["capacity"]))
        errors = confusion_metrics(test_y, flags, 5.0)
        top_rows.append({"model": model, **record, **{f"test_{k}": v for k, v in errors.items()},
                         "test_k": top_meta["k"], "test_boundary_probability": top_meta["boundary_probability"],
                         "test_records_tied_at_boundary": top_meta["records_tied_at_boundary"]})
    pd.DataFrame(top_rows).to_csv(out / "decisions/top_k_test_results.csv", index=False)

    test_metrics = pd.read_csv(out / "statistics/final_test_metrics.csv").to_dict("records")
    reliability_rows = pd.read_csv(out / "calibration/final_test_reliability_bins.csv").to_dict("records")
    curve_rows = pd.read_csv(out / "statistics/test_curve_points.csv").to_dict("records")
    create_figures(test_predictions, test_y, test_metrics, reliability_rows, curve_rows,
                   sensitivity_rows, slice_rows, overlap_rows, out)

    b_reps = pd.read_csv(out / "statistics/bootstrap_replicates.csv")
    if len(b_reps) != BOOTSTRAP_N * len(MODEL_NAMES) * 2:
        raise RuntimeError("Bootstrap replicate table does not contain all model/score rows")
    one_class_replicates = int((~b_reps[(b_reps.model == MODEL_NAMES[0]) & (b_reps.probability_type == "raw")].ranking_metrics_valid).sum())
    with np.load(out / "statistics/bootstrap_indices.npz") as bootstrap:
        if bootstrap["indices"].shape != (BOOTSTRAP_N, len(test_idx)) or int(bootstrap["seed"]) != BOOTSTRAP_SEED:
            raise RuntimeError("Saved paired bootstrap indices do not match frozen dimensions/seed")
    fit_times = cv.groupby("model").fit_seconds.sum().to_dict()
    report = {"status": "Phase 2 core experiment and reports complete; independent review pending",
              "protocol_sha256": freeze["protocol_sha256"], "split_sha256": freeze["split_artifact_sha256"],
              "source_sha256": freeze["source_archive_sha256"], "cv_fit_seconds_by_model": fit_times,
              "cv_fit_seconds_total": float(sum(fit_times.values())), "cv_config_fold_fits": int(len(cv)),
              "cv_failures": int((cv.status != "ok").sum()),
              "convergence_warning_count": int(cv.convergence_warning_count.sum()),
              "bootstrap_replicates": BOOTSTRAP_N, "bootstrap_seed": BOOTSTRAP_SEED,
              "one_class_bootstrap_resamples": one_class_replicates,
              "python": __import__("sys").version, "pandas": pd.__version__, "numpy": np.__version__,
              "scikit_learn": sklearn.__version__, "wall_seconds_for_report_finalization": time.perf_counter() - started,
              "model_training_performed_by_this_finalizer": False}
    (out / "phase2_execution_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    print(json.dumps(finalize_reports(), indent=2))
