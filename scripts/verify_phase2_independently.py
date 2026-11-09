"""Independent numerical audit of saved Phase 2 outputs; never fits models."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TARGET = "default payment next month"
MODELS = ["Logistic Regression", "Random Forest", "HistGradientBoosting"]
PAY_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
FEATURES = ["LIMIT_BAL", *PAY_COLUMNS, *[f"BILL_AMT{i}" for i in range(1, 7)], *[f"PAY_AMT{i}" for i in range(1, 7)]]
METRICS = ["average_precision", "roc_auc", "brier_score", "ece"]
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 2026
TOLERANCE = 1e-12
EXPECTED_SLICE_TPS = {
    "all_six_PAY_nonpositive": {"Logistic Regression": 0, "Random Forest": 0, "HistGradientBoosting": 0},
    "PAY_0_equals_zero": {"Logistic Regression": 0, "Random Forest": 1, "HistGradientBoosting": 0},
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_probability(values, name: str = "probability") -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    if ((array < 0.0) | (array > 1.0)).any():
        raise ValueError(f"{name} contains values outside [0, 1]")
    return array


def validate_prediction_alignment(predictions: pd.DataFrame, expected_positions,
                                  expected_ids, expected_labels) -> None:
    required = {"row_position", "ID", "y_true", "split"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing required columns: {sorted(missing)}")
    positions = np.asarray(expected_positions, dtype=np.int64)
    ids = np.asarray(expected_ids)
    labels = np.asarray(expected_labels, dtype=np.int64)
    if len(predictions) != len(positions):
        raise ValueError(f"Prediction row count {len(predictions)} != frozen test size {len(positions)}")
    got_positions = predictions["row_position"].to_numpy(dtype=np.int64)
    got_ids = predictions["ID"].to_numpy()
    got_labels = predictions["y_true"].to_numpy(dtype=np.int64)
    if pd.Series(got_positions).duplicated().any():
        raise ValueError("Duplicate row_position in prediction table")
    if pd.Series(got_ids).duplicated().any():
        raise ValueError("Duplicate customer ID in prediction table")
    if len(np.unique(positions)) != len(positions):
        raise ValueError("Frozen test indices contain duplicate row positions")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Source Test records contain duplicate IDs")
    if not np.array_equal(got_positions, positions):
        raise ValueError("Prediction row positions are missing, extra, or misordered")
    if not np.array_equal(got_ids, ids):
        raise ValueError("Prediction IDs do not align with frozen Test rows")
    if not np.array_equal(got_labels, labels):
        raise ValueError("Saved prediction labels do not match source labels at frozen Test rows")
    if set(predictions["split"].astype(str)) != {"final_test"}:
        raise ValueError("Prediction table contains a split label other than final_test")
    if not set(np.unique(got_labels)).issubset({0, 1}):
        raise ValueError("Test labels are not binary")


def independent_ece(labels, probabilities, bin_count: int = 10) -> tuple[float, list[dict]]:
    y = np.asarray(labels, dtype=np.int64)
    p = require_probability(probabilities)
    if y.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("ECE requires nonempty aligned labels and probabilities")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("ECE labels must be binary")
    edges = np.arange(bin_count + 1, dtype=np.float64) / bin_count
    membership = np.searchsorted(edges, p, side="right") - 1
    membership = np.minimum(np.maximum(membership, 0), bin_count - 1)
    output = []
    total_error = 0.0
    for number in range(bin_count):
        selected = membership == number
        count = int(selected.sum())
        if count:
            average_score = float(np.mean(p[selected], dtype=np.float64))
            empirical_rate = float(np.mean(y[selected], dtype=np.float64))
            total_error += (count / len(y)) * abs(average_score - empirical_rate)
        else:
            average_score = None
            empirical_rate = None
        output.append({"bin": number, "lower_inclusive": float(edges[number]),
                       "upper_exclusive": float(edges[number + 1]) if number < bin_count - 1 else None,
                       "count": count, "mean_probability": average_score,
                       "observed_positive_rate": empirical_rate})
    return float(total_error), output


def independent_metrics(labels, probabilities) -> tuple[dict, list[dict]]:
    y = np.asarray(labels, dtype=np.int64)
    p = require_probability(probabilities)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("Metric inputs must be nonempty and aligned")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("AP and ROC-AUC verification requires both classes")
    ece_value, bin_rows = independent_ece(y, p, 10)
    return ({"average_precision": float(average_precision_score(y, p)),
             "roc_auc": float(roc_auc_score(y, p)),
             "brier_score": float(brier_score_loss(y, p)),
             "ece": ece_value}, bin_rows)


def independent_confusion(labels, flags, fn_cost: float = 5.0, fp_cost: float = 1.0) -> dict:
    y = np.asarray(labels, dtype=np.int64)
    f = np.asarray(flags, dtype=bool)
    if y.ndim != 1 or f.ndim != 1 or len(y) != len(f) or len(y) == 0:
        raise ValueError("Confusion counts require nonempty aligned labels and flags")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Confusion labels must be binary")
    tp = int(np.count_nonzero((y == 1) & f))
    fp = int(np.count_nonzero((y == 0) & f))
    tn = int(np.count_nonzero((y == 0) & ~f))
    fn = int(np.count_nonzero((y == 1) & ~f))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "false_positive_rate": fpr,
            "flagged_n": int(f.sum()), "flagged_fraction": float(np.mean(f)),
            "hypothetical_loss": float(fp_cost * fp + fn_cost * fn)}


def recompute_bootstrap_rows(labels, probability_columns: dict[str, np.ndarray], indices) -> list[dict]:
    y = np.asarray(labels, dtype=np.int64)
    draws = np.asarray(indices)
    if draws.ndim != 2 or draws.shape[1] != len(y) or not np.issubdtype(draws.dtype, np.integer):
        raise ValueError("Bootstrap indices must be an integer matrix with one row-sized draw per replicate")
    if (draws < 0).any() or (draws >= len(y)).any():
        raise ValueError("Bootstrap index is outside the Test row range")
    scores = {key: require_probability(value, key) for key, value in probability_columns.items()}
    if any(len(value) != len(y) for value in scores.values()):
        raise ValueError("Bootstrap score vectors are not aligned to Test labels")
    records = []
    for replicate, draw in enumerate(draws):
        sampled_y = y[draw]
        ranking_valid = len(np.unique(sampled_y)) == 2
        for key, probability in scores.items():
            sampled_probability = probability[draw]
            ece_value, _ = independent_ece(sampled_y, sampled_probability, 10)
            records.append({"replicate": replicate, "model_probability": key,
                            "average_precision": float(average_precision_score(sampled_y, sampled_probability)) if ranking_valid else None,
                            "roc_auc": float(roc_auc_score(sampled_y, sampled_probability)) if ranking_valid else None,
                            "brier_score": float(brier_score_loss(sampled_y, sampled_probability)),
                            "ece": ece_value, "ranking_metrics_valid": ranking_valid})
    return records


def compare_bootstrap_replicates(recomputed: list[dict], saved: pd.DataFrame,
                                 tolerance: float = TOLERANCE) -> list[dict]:
    key_columns = ["replicate", "model", "probability_type"]
    needed = set(key_columns + METRICS + ["ranking_metrics_valid"])
    if needed - set(saved.columns):
        raise ValueError(f"Saved bootstrap table missing columns: {sorted(needed - set(saved.columns))}")
    new = pd.DataFrame(recomputed).copy()
    new[["model", "probability_type"]] = new["model_probability"].str.rsplit("__", n=1, expand=True)
    old = saved.copy()
    if old.duplicated(key_columns).any() or new.duplicated(key_columns).any():
        raise ValueError("Duplicate model/replicate bootstrap keys")
    joined = old.merge(new, on=key_columns, how="outer", suffixes=("_saved", "_recomputed"), indicator=True)
    if not (joined["_merge"] == "both").all():
        raise ValueError("Saved and recomputed bootstrap tables have different paired replicate keys")
    comparisons = []
    for _, row in joined.iterrows():
        for metric in METRICS:
            left, right = row[f"{metric}_saved"], row[f"{metric}_recomputed"]
            if pd.isna(left) and pd.isna(right):
                difference, passed = None, True
            elif pd.isna(left) or pd.isna(right):
                difference, passed = None, False
            else:
                difference = float(right - left)
                passed = abs(difference) <= tolerance
            comparisons.append({"replicate": int(row["replicate"]), "model": row["model"],
                                "probability_type": row["probability_type"], "metric": metric,
                                "saved": left, "recomputed": right, "difference": difference,
                                "passed": bool(passed)})
        if bool(row["ranking_metrics_valid_saved"]) != bool(row["ranking_metrics_valid_recomputed"]):
            raise ValueError("Ranking-metric validity differs between saved and recomputed bootstrap rows")
    failed = [item for item in comparisons if not item["passed"]]
    if failed:
        first = failed[0]
        raise ValueError(f"Bootstrap values are not paired/reproducible; first mismatch: {first}")
    return comparisons


def load_source_workbook() -> pd.DataFrame:
    archive = ROOT / "data/raw/uci_default_credit_card_350.zip"
    with zipfile.ZipFile(archive) as package:
        excel_files = [name for name in package.namelist() if name.lower().endswith(".xls")]
        if len(excel_files) != 1:
            raise ValueError(f"Expected one original workbook; found {excel_files}")
        with package.open(excel_files[0]) as stream:
            data = pd.read_excel(stream, header=1, engine="xlrd")
    data.columns = [str(column).strip() for column in data.columns]
    if len(data) != 30000 or TARGET not in data:
        raise ValueError("Source workbook row count or target column does not match the audit")
    return data


def compare_float(saved, recomputed, tolerance=TOLERANCE) -> dict:
    difference = float(recomputed - saved)
    return {"saved": float(saved), "recomputed": float(recomputed), "difference": difference,
            "absolute_tolerance": tolerance, "passed": abs(difference) <= tolerance,
            "difference_class": "numerical_rounding_within_tolerance" if abs(difference) <= tolerance else "discrepancy"}


def run_verification() -> dict:
    started = time.perf_counter()
    verification_dir = ROOT / "results/verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((ROOT / "results/protocol_v1.0_freeze.json").read_text())
    protocol_path = ROOT / "docs/evaluation_protocol_v1.0.md"
    protocol_hash = file_sha256(protocol_path)
    protocol_hash_ok = protocol_hash == frozen["protocol_sha256"]
    if not protocol_hash_ok:
        raise ValueError("Frozen protocol SHA-256 differs from the freeze manifest")

    split_file = ROOT / "results/audit/proposed_splits.json"
    if file_sha256(split_file) != frozen["split_artifact_sha256"]:
        raise ValueError("Frozen split artifact SHA-256 differs from the freeze manifest")
    split = json.loads(split_file.read_text())
    test_positions = np.asarray(split["indices_zero_based"]["final_test"], dtype=np.int64)
    train_positions = np.asarray(split["indices_zero_based"]["train_cv"], dtype=np.int64)
    if len(test_positions) != 4500 or len(np.unique(test_positions)) != 4500:
        raise ValueError("Frozen Test indices are missing, duplicated, or the wrong size")

    source = load_source_workbook()
    source_test = source.iloc[test_positions]
    predictions_path = ROOT / "results/predictions/final_test_predictions.csv"
    predictions = pd.read_csv(predictions_path)
    validate_prediction_alignment(predictions, test_positions,
                                  source_test["ID"].to_numpy(), source_test[TARGET].to_numpy())
    labels = predictions["y_true"].to_numpy(dtype=np.int64)
    positive_n = int(np.count_nonzero(labels == 1))
    negative_n = int(np.count_nonzero(labels == 0))
    if positive_n != 995 or negative_n != 3505:
        raise ValueError(f"Expected exactly 995 positive and 3,505 negative Test labels; got {positive_n}/{negative_n}")

    probability_map = {}
    for model in MODELS:
        for probability_type in ["raw", "sigmoid"]:
            column = f"{model}__{probability_type}"
            if column not in predictions:
                raise ValueError(f"Missing saved score column: {column}")
            probability_map[(model, probability_type)] = require_probability(predictions[column], column)

    # Recompute all Final Test metrics directly from row-level predictions.
    original_metrics = pd.read_csv(ROOT / "results/statistics/final_test_metrics.csv")
    metric_rows, reliability_rows = [], []
    metric_checks = []
    for model in MODELS:
        for probability_type in ["raw", "sigmoid"]:
            values, bins = independent_metrics(labels, probability_map[(model, probability_type)])
            row = original_metrics[(original_metrics.model == model) &
                                   (original_metrics.probability_type == probability_type)]
            if len(row) != 1:
                raise ValueError(f"Expected one saved metric row for {model}/{probability_type}; got {len(row)}")
            saved = row.iloc[0]
            metric_rows.append({"model": model, "probability_type": probability_type, **values})
            for bin_row in bins:
                reliability_rows.append({"model": model, "probability_type": probability_type, **bin_row})
            for metric in METRICS:
                metric_checks.append({"model": model, "probability_type": probability_type, "metric": metric,
                                      **compare_float(float(saved[metric]), values[metric])})
    pd.DataFrame(metric_rows).to_csv(verification_dir / "recomputed_test_metrics.csv", index=False, float_format="%.17g")
    pd.DataFrame(metric_checks).to_csv(verification_dir / "reported_metric_comparison.csv", index=False, float_format="%.17g")
    pd.DataFrame(reliability_rows).to_csv(verification_dir / "independent_reliability_bins.csv", index=False, float_format="%.17g")

    # Apply only the previously locked probabilities and thresholds.
    locks = json.loads((ROOT / "results/decisions/locked_decisions.json").read_text())
    original_test_metrics = original_metrics[original_metrics.probability_type == "locked_threshold_test"]
    threshold_rows, threshold_checks = [], []
    for model in MODELS:
        if model not in locks.get("primary", {}):
            raise ValueError(f"No locked primary threshold stored for {model}")
        threshold = float(locks["primary"][model]["threshold"])
        scores = probability_map[(model, "sigmoid")]
        result = independent_confusion(labels, scores >= threshold, fn_cost=5.0, fp_cost=1.0)
        result["threshold"] = threshold
        threshold_rows.append({"model": model, **result})
        original = original_test_metrics[original_test_metrics.model == model]
        if len(original) != 1:
            raise ValueError(f"Expected one original locked-threshold test row for {model}")
        saved = original.iloc[0]
        if float(saved["threshold"]) != threshold:
            raise ValueError(f"Saved Test threshold for {model} differs from locked threshold")
        for key in ["tp", "fp", "tn", "fn", "flagged_n"]:
            threshold_checks.append({"model": model, "field": key, "saved": int(saved[key]),
                                     "recomputed": int(result[key]), "difference": int(result[key] - saved[key]),
                                     "passed": int(result[key]) == int(saved[key])})
        for key in ["precision", "recall", "false_positive_rate", "flagged_fraction", "hypothetical_loss"]:
            threshold_checks.append({"model": model, "field": key, **compare_float(float(saved[key]), float(result[key]))})
    pd.DataFrame(threshold_rows).to_csv(verification_dir / "recomputed_primary_thresholds.csv", index=False, float_format="%.17g")
    pd.DataFrame(threshold_checks).to_csv(verification_dir / "primary_threshold_comparison.csv", index=False, float_format="%.17g")

    original_sensitivity = pd.read_csv(ROOT / "results/decisions/cost_capacity_sensitivity.csv")
    sensitivity_rows, sensitivity_checks = [], []
    locked_sensitivity = locks.get("sensitivity", [])
    if len(locked_sensitivity) != 27 or len(original_sensitivity) != 27:
        raise ValueError("Expected 27 saved locked sensitivity decisions and 27 reported test rows")
    for decision in locked_sensitivity:
        model = decision["model"]
        capacity = float(decision["capacity"])
        ratio = float(decision["fn_to_fp_ratio"])
        matches = original_sensitivity[(original_sensitivity.model == model) &
                                       np.isclose(original_sensitivity.capacity, capacity) &
                                       np.isclose(original_sensitivity.fn_to_fp_ratio, ratio)]
        if len(matches) != 1:
            raise ValueError(f"Sensitivity result row is not unique for {model}/{capacity}/{ratio}")
        threshold = float(decision["threshold"])
        result = independent_confusion(labels, probability_map[(model, "sigmoid")] >= threshold,
                                       fn_cost=ratio, fp_cost=1.0)
        saved = matches.iloc[0]
        record = {"model": model, "capacity": capacity, "fn_to_fp_ratio": ratio, "locked_threshold": threshold, **result}
        sensitivity_rows.append(record)
        for key in ["tp", "fp", "tn", "fn", "flagged_n"]:
            field = f"test_{key}"
            sensitivity_checks.append({"model": model, "capacity": capacity, "fn_to_fp_ratio": ratio,
                                       "field": field, "saved": int(saved[field]), "recomputed": result[key],
                                       "difference": int(result[key] - saved[field]), "passed": int(result[key]) == int(saved[field])})
        for key in ["flagged_fraction", "hypothetical_loss"]:
            field = f"test_{key}"
            sensitivity_checks.append({"model": model, "capacity": capacity, "fn_to_fp_ratio": ratio,
                                       "field": field, **compare_float(float(saved[field]), float(result[key]))})
    pd.DataFrame(sensitivity_rows).to_csv(verification_dir / "recomputed_sensitivity.csv", index=False, float_format="%.17g")
    pd.DataFrame(sensitivity_checks).to_csv(verification_dir / "sensitivity_comparison.csv", index=False, float_format="%.17g")

    # Recompute bootstrap replicate metrics on the exact preserved draws.
    bootstrap_path = ROOT / "results/statistics/bootstrap_indices.npz"
    with np.load(bootstrap_path) as saved_indices:
        if not {"indices", "seed", "test_row_positions"}.issubset(saved_indices.files):
            raise ValueError("Saved bootstrap index archive is missing indices, seed, or Test row positions")
        draws = saved_indices["indices"].copy()
        seed = int(saved_indices["seed"])
        saved_bootstrap_positions = saved_indices["test_row_positions"].copy()
    if seed != BOOTSTRAP_SEED or draws.shape != (N_BOOTSTRAP, len(labels)):
        raise ValueError("Saved bootstrap seed or dimensions differ from the frozen specification")
    if not np.array_equal(saved_bootstrap_positions, test_positions):
        raise ValueError("Bootstrap Test row positions differ from the frozen Test order")
    expected_draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(labels), size=(N_BOOTSTRAP, len(labels)), dtype=np.int16)
    seed_sequence_matches = bool(np.array_equal(draws, expected_draws))
    probability_columns = {f"{model}__{kind}": probability_map[(model, kind)]
                           for model in MODELS for kind in ["raw", "sigmoid"]}
    recomputed_bootstrap = recompute_bootstrap_rows(labels, probability_columns, draws)
    saved_replicates = pd.read_csv(ROOT / "results/statistics/bootstrap_replicates.csv")
    replicate_comparisons = compare_bootstrap_replicates(recomputed_bootstrap, saved_replicates)
    pd.DataFrame(recomputed_bootstrap).assign(
        model=lambda frame: frame.model_probability.str.rsplit("__", n=1).str[0],
        probability_type=lambda frame: frame.model_probability.str.rsplit("__", n=1).str[1]
    ).drop(columns=["model_probability"]).to_csv(
        verification_dir / "recomputed_bootstrap_replicates.csv", index=False, float_format="%.17g")
    pd.DataFrame(replicate_comparisons).to_csv(
        verification_dir / "bootstrap_replicate_comparison.csv", index=False, float_format="%.17g")

    point_metrics = {(row["model"], row["probability_type"]): row for row in metric_rows}
    replicate_frame = pd.DataFrame(recomputed_bootstrap)
    interval_comparisons = []
    original_intervals = pd.read_csv(ROOT / "results/statistics/bootstrap_intervals.csv")
    for kind in ["raw", "sigmoid"]:
        for metric in METRICS:
            per_model_values = {}
            for model in MODELS:
                sample = replicate_frame[(replicate_frame.model_probability == f"{model}__{kind}")]
                values = sample[metric].dropna().to_numpy(dtype=np.float64)
                per_model_values[model] = values
                valid_n = len(values)
                old = original_intervals[(original_intervals.model == model) &
                                         (original_intervals.probability_type == kind) &
                                         (original_intervals.metric == metric)]
                if len(old) != 1:
                    raise ValueError(f"Expected one interval record for {model}/{kind}/{metric}")
                saved = old.iloc[0]
                point = point_metrics[(model, kind)][metric]
                lower, upper = np.percentile(values, [2.5, 97.5])
                for field, value in [("test_estimate", point), ("ci_95_low", lower), ("ci_95_high", upper)]:
                    interval_comparisons.append({"model": model, "probability_type": kind, "metric": metric,
                                                 "comparison": field, **compare_float(float(saved[field]), float(value))})
                interval_comparisons.append({"model": model, "probability_type": kind, "metric": metric,
                                             "comparison": "valid_replicates", "saved": int(saved.valid_replicates),
                                             "recomputed": int(valid_n), "difference": int(valid_n - saved.valid_replicates),
                                             "passed": int(valid_n) == int(saved.valid_replicates)})
            baseline = per_model_values["Logistic Regression"]
            for model in MODELS:
                deltas = per_model_values[model] - baseline
                original = original_intervals[(original_intervals.model == model) &
                                              (original_intervals.probability_type == kind) &
                                              (original_intervals.metric == metric)].iloc[0]
                difference_point = (point_metrics[(model, kind)][metric] -
                                    point_metrics[("Logistic Regression", kind)][metric]) if model != "Logistic Regression" else 0.0
                dlow, dhigh = np.percentile(deltas, [2.5, 97.5])
                for field, value in [("paired_difference_vs_logistic", difference_point),
                                     ("paired_difference_ci_95_low", dlow),
                                     ("paired_difference_ci_95_high", dhigh)]:
                    interval_comparisons.append({"model": model, "probability_type": kind, "metric": metric,
                                                 "comparison": field, **compare_float(float(original[field]), float(value))})
    pd.DataFrame(interval_comparisons).to_csv(
        verification_dir / "bootstrap_interval_comparison.csv", index=False, float_format="%.17g")

    # Verify reported behavior slices from source features, frozen positions, labels, and saved scores.
    test_features = source_test[FEATURES].to_numpy()
    pay = source_test[PAY_COLUMNS].to_numpy(dtype=np.float64)
    nonpositive_history = np.all(pay <= 0, axis=1)
    pay0_zero = source_test["PAY_0"].to_numpy() == 0
    slice_definitions = {"all_six_PAY_nonpositive": nonpositive_history, "PAY_0_equals_zero": pay0_zero}
    existing_slices = pd.read_csv(ROOT / "results/slices/final_test_behavioral_slices.csv")
    slice_rows, slice_checks = [], []
    for slice_name, mask in slice_definitions.items():
        subset_y = labels[mask]
        n_rows, n_positive = int(mask.sum()), int(subset_y.sum())
        if slice_name == "all_six_PAY_nonpositive":
            expected_n, expected_pos = 3042, 357
            saved_rows = existing_slices[(existing_slices["slice"] == "Any PAY code > 0") &
                                         (existing_slices["group"] == "false")]
        else:
            expected_n, expected_pos = 2229, 285
            saved_rows = existing_slices[(existing_slices["slice"] == "PAY_0 grouped") &
                                         (existing_slices["group"] == "0")]
        for name, observed, expected in [("record_count", n_rows, expected_n), ("positive_count", n_positive, expected_pos)]:
            slice_checks.append({"slice": slice_name, "check": name, "expected": expected,
                                 "recomputed": observed, "passed": observed == expected})
        for model in MODELS:
            scores = probability_map[(model, "sigmoid")][mask]
            stats = independent_metrics(subset_y, scores)[0]
            threshold = float(locks["primary"][model]["threshold"])
            errors = independent_confusion(subset_y, scores >= threshold)
            slice_rows.append({"slice": slice_name, "model": model, "n": n_rows, "positive": n_positive,
                               "negative": int(n_rows - n_positive), **stats, "threshold": threshold, **errors})
            expected_tp = EXPECTED_SLICE_TPS[slice_name][model]
            slice_checks.append({"slice": slice_name, "check": f"{model}_expected_TP_at_locked_threshold",
                                 "expected": expected_tp, "recomputed": errors["tp"],
                                 "passed": errors["tp"] == expected_tp})
            matching = saved_rows[saved_rows.model == model]
            if len(matching) != 1:
                raise ValueError(f"Missing saved slice row for {slice_name}/{model}")
            old = matching.iloc[0]
            expected_counts = {"n": n_rows, "positive": n_positive, "negative": n_rows - n_positive,
                               "tp": errors["tp"], "fp": errors["fp"], "tn": errors["tn"], "fn": errors["fn"]}
            for field, recomputed_count in expected_counts.items():
                saved_field = f"threshold_{field}" if field in {"tp", "fp", "tn", "fn"} else field
                slice_checks.append({"slice": slice_name, "check": f"{model}_saved_{field}",
                                     "expected": int(old[saved_field]), "recomputed": int(recomputed_count),
                                     "passed": int(old[saved_field]) == int(recomputed_count)})
    pd.DataFrame(slice_rows).to_csv(verification_dir / "recomputed_critical_slices.csv", index=False, float_format="%.17g")
    pd.DataFrame(slice_checks).to_csv(verification_dir / "critical_slice_comparison.csv", index=False, float_format="%.17g")

    train_features = source.iloc[train_positions][FEATURES].to_numpy()
    train_profiles = {tuple(row) for row in train_features.tolist()}
    overlap_mask = np.asarray([tuple(row) in train_profiles for row in test_features.tolist()], dtype=bool)
    overlap_n = int(overlap_mask.sum())
    saved_overlap = json.loads((ROOT / "results/diagnostics/duplicate_profile_overlap.json").read_text())["final_test"]
    overlap_verification = {"test_n": int(len(overlap_mask)), "overlap_n": overlap_n,
                            "overlap_fraction": float(overlap_mask.mean()),
                            "nonoverlap_n": int((~overlap_mask).sum()),
                            "overlap_positive_n": int(labels[overlap_mask].sum()),
                            "saved_overlap_n": int(saved_overlap["overlap_n"]),
                            "saved_overlap_fraction": float(saved_overlap["overlap_fraction"]),
                            "overlap_count_matches": overlap_n == int(saved_overlap["overlap_n"]),
                            "fraction_matches": abs(float(overlap_mask.mean()) - float(saved_overlap["overlap_fraction"])) <= TOLERANCE,
                            "interpretation": "Exact feature-profile overlap does not prove duplicate persons."}
    (verification_dir / "duplicate_profile_overlap_verification.json").write_text(json.dumps(overlap_verification, indent=2))

    # Confirm that the two main model-ranking claims follow from recomputed values.
    calibrated_metrics = {model: point_metrics[(model, "sigmoid")] for model in MODELS}
    raw_metrics = {model: point_metrics[(model, "raw")] for model in MODELS}
    threshold_metrics = {row["model"]: row for row in threshold_rows}
    claim_checks = {
        "calibration_not_universally_beneficial": {
            model: {"brier_raw": raw_metrics[model]["brier_score"],
                    "brier_sigmoid": calibrated_metrics[model]["brier_score"],
                    "brier_change": calibrated_metrics[model]["brier_score"] - raw_metrics[model]["brier_score"],
                    "ece_raw": raw_metrics[model]["ece"], "ece_sigmoid": calibrated_metrics[model]["ece"],
                    "ece_change": calibrated_metrics[model]["ece"] - raw_metrics[model]["ece"]}
            for model in MODELS},
        "HGB_highest_AP": max(MODELS, key=lambda model: calibrated_metrics[model]["average_precision"]) == "HistGradientBoosting",
        "HGB_highest_ROC_AUC": max(MODELS, key=lambda model: calibrated_metrics[model]["roc_auc"]) == "HistGradientBoosting",
        "RF_lowest_primary_test_loss": min(MODELS, key=lambda model: threshold_metrics[model]["hypothetical_loss"]) == "Random Forest",
        "RF_minus_HGB_primary_test_loss": threshold_metrics["Random Forest"]["hypothetical_loss"] - threshold_metrics["HistGradientBoosting"]["hypothetical_loss"],
        "primary_thresholds": {model: {"threshold": locks["primary"][model]["threshold"],
                                       "test_loss": threshold_metrics[model]["hypothetical_loss"]}
                               for model in MODELS},
    }
    (verification_dir / "scientific_claim_checks.json").write_text(json.dumps(claim_checks, indent=2))

    # Independent CV aggregation confirms the stored AP-based selection without fitting.
    cv = pd.read_csv(ROOT / "results/cv/cv_fold_results.csv")
    cv_checks = []
    selected = json.loads((ROOT / "results/cv/selected_models.json").read_text())
    for model in MODELS:
        block = cv[cv.model == model]
        if len(block) != 60 or set(block.status) != {"ok"}:
            raise ValueError(f"Incomplete saved CV records for {model}")
        grouped = block.groupby("config_index").average_precision.agg(["mean", "count"])
        if len(grouped) != 12 or not (grouped["count"] == 5).all():
            raise ValueError(f"Saved CV fold structure invalid for {model}")
        best_config = int(grouped["mean"].sort_values(ascending=False, kind="stable").index[0])
        best_mean = float(grouped.loc[best_config, "mean"])
        original = selected[model]
        cv_checks.append({"model": model, "best_config_recomputed": best_config,
                          "best_config_saved": int(original["config_index"]),
                          "mean_AP_recomputed": best_mean, "mean_AP_saved": float(original["mean_cv_ap"]),
                          "passed": best_config == int(original["config_index"]) and
                                   abs(best_mean - float(original["mean_cv_ap"])) <= TOLERANCE})
    (verification_dir / "cv_selection_verification.json").write_text(json.dumps(cv_checks, indent=2))

    all_float_checks = metric_checks + threshold_checks + sensitivity_checks + interval_comparisons
    failed_float_checks = [row for row in all_float_checks if not row["passed"]]
    failed_bootstrap = [row for row in replicate_comparisons if not row["passed"]]
    failed_slices = [row for row in slice_checks if not row["passed"]]
    failed_cv = [row for row in cv_checks if not row["passed"]]
    original_sensitivity_thresholds_present = len(locked_sensitivity) == 27 and all("threshold" in row for row in locked_sensitivity)
    report = {
        "verdict": "PASS" if not any([failed_float_checks, failed_bootstrap, failed_slices, failed_cv]) else "FAIL",
        "scope": "Independent verification of saved Phase 2 evidence; no training, refitting, threshold search, or source-result modification",
        "verified_at_runtime": {"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__,
                                "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "source_artifacts": {
            "protocol": "docs/evaluation_protocol_v1.0.md", "protocol_sha256": protocol_hash,
            "protocol_hash_matches_freeze": protocol_hash_ok,
            "split_indices": "results/audit/proposed_splits.json", "split_sha256": file_sha256(split_file),
            "test_predictions": "results/predictions/final_test_predictions.csv",
            "locked_decisions": "results/decisions/locked_decisions.json",
            "reported_metrics": "results/statistics/final_test_metrics.csv",
            "bootstrap_indices": "results/statistics/bootstrap_indices.npz",
            "bootstrap_replicates": "results/statistics/bootstrap_replicates.csv",
            "reported_intervals": "results/statistics/bootstrap_intervals.csv",
            "sensitivity_test_results": "results/decisions/cost_capacity_sensitivity.csv",
            "CV_results": "results/cv/cv_fold_results.csv",
            "source_workbook": "data/raw/uci_default_credit_card_350.zip::default of credit card clients.xls",
        },
        "test_alignment": {"rows": int(len(labels)), "positive": positive_n, "negative": negative_n,
                           "row_position_order_matches_frozen_test": True, "IDs_unique_and_aligned": True,
                           "labels_match_original_workbook": True, "scores_finite_in_unit_interval": True},
        "tolerance": {"absolute": TOLERANCE,
                      "interpretation": "Differences within 1e-12 are treated as floating-point/serialization roundoff; larger differences are discrepancies."},
        "bootstrap": {"replicates": int(len(draws)), "seed": seed,
                      "saved_indices_match_default_rng_sequence": seed_sequence_matches,
                      "same_indices_used_for_all_models": True,
                      "one_class_replicates": int(np.count_nonzero(~replicate_frame.groupby("replicate").ranking_metrics_valid.first().to_numpy())),
                      "all_saved_replicate_values_match_recomputation": not failed_bootstrap,
                      "interval_comparisons_passed": not [row for row in interval_comparisons if not row["passed"]]},
        "thresholds": {"saved_thresholds_used_without_optimization": True,
                       "all_27_sensitivity_rows_recomputed": len(sensitivity_rows) == 27,
                       "all_threshold_comparisons_passed": not [row for row in threshold_checks + sensitivity_checks if not row["passed"]]},
        "critical_slices": {"checks": len(slice_checks), "all_claim_checks_passed": not failed_slices},
        "duplicate_profile_overlap": overlap_verification,
        "cv_selection": {"models_checked": len(cv_checks), "all_saved_selections_reproduced": not failed_cv},
        "numerical_comparison": {"float_checks": len(all_float_checks), "failed_float_checks": len(failed_float_checks),
                                 "failed_bootstrap_checks": len(failed_bootstrap),
                                 "failed_slice_checks": len(failed_slices), "failed_cv_checks": len(failed_cv)},
        "duration_seconds": time.perf_counter() - started,
    }
    (verification_dir / "verification_manifest.json").write_text(json.dumps(report, indent=2))
    if report["verdict"] != "PASS":
        raise RuntimeError(f"Independent verification found discrepancies; inspect {verification_dir}")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_verification()
