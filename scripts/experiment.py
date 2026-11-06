"""Run the frozen DS-A4 Phase 2 experiment and write reproducible artifacts."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import pickle
import platform
import sys
import time
import warnings
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xlrd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_curve, roc_auc_score,
                             roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/raw/uci_default_credit_card_350.zip"
SPLITS_PATH = ROOT / "results/audit/proposed_splits.json"
PROTOCOL_PATH = ROOT / "docs/evaluation_protocol_v1.0.md"
FREEZE_PATH = ROOT / "results/protocol_v1.0_freeze.json"
TARGET = "default payment next month"
PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
NUMERIC = ["LIMIT_BAL"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]
FEATURES = ["LIMIT_BAL"] + PAY + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]
DEMOGRAPHICS = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]
MODEL_NAMES = ["Logistic Regression", "Random Forest", "HistGradientBoosting"]
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 2026
METRIC_NAMES = ["average_precision", "roc_auc", "brier_score", "ece"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_frozen_inputs() -> tuple[dict, dict]:
    freeze = json.loads(FREEZE_PATH.read_text())
    if sha256(PROTOCOL_PATH) != freeze["protocol_sha256"]:
        raise RuntimeError("Frozen protocol SHA-256 mismatch; stop before any model fit")
    if sha256(SPLITS_PATH) != freeze["split_artifact_sha256"]:
        raise RuntimeError("Frozen split artifact SHA-256 mismatch; stop before any model fit")
    if sha256(ARCHIVE) != freeze["source_archive_sha256"]:
        raise RuntimeError("Source archive SHA-256 mismatch; stop before any model fit")
    split_data = json.loads(SPLITS_PATH.read_text())
    expected_sizes = {"train_cv": 18000, "calibration": 4500, "threshold_selection": 3000, "final_test": 4500}
    if {k: len(v) for k, v in split_data["indices_zero_based"].items()} != expected_sizes:
        raise RuntimeError("Frozen split sizes do not match protocol")
    return freeze, split_data


def load_data() -> pd.DataFrame:
    with zipfile.ZipFile(ARCHIVE) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".xls")]
        if members != ["default of credit card clients.xls"]:
            raise RuntimeError(f"Unexpected original workbook inventory: {members}")
        with zf.open(members[0]) as stream:
            frame = pd.read_excel(stream, header=1, engine="xlrd")
    frame.columns = [str(c).strip() for c in frame.columns]
    if len(frame) != 30000 or TARGET not in frame or frame[FEATURES + DEMOGRAPHICS + ["ID", TARGET]].shape[1] != 25:
        raise RuntimeError("Dataset schema differs from the audited source")
    return frame


def make_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("pay_states", categorical_pipe, PAY),
    ], remainder="drop", sparse_threshold=0.0, verbose_feature_names_out=True)


def configurations() -> dict[str, list[dict]]:
    return {
        "Logistic Regression": [
            {"C": c, "class_weight": w}
            for c, w in itertools.product([0.01, 0.03, 0.1, 0.3, 1.0, 3.0], [None, "balanced"])
        ],
        "Random Forest": [
            {"max_depth": d, "min_samples_leaf": leaf, "max_features": features}
            for d, leaf, features in itertools.product([8, 16, None], [1, 5], ["sqrt", 0.7])
        ],
        "HistGradientBoosting": [
            {"learning_rate": rate, "max_leaf_nodes": leaves, "l2_regularization": l2}
            for rate, leaves, l2 in itertools.product([0.03, 0.1, 0.2], [15, 31], [0.0, 1.0])
        ],
    }


def estimator(model_name: str, params: dict) -> Pipeline:
    if model_name == "Logistic Regression":
        model = LogisticRegression(solver="lbfgs", max_iter=5000, random_state=42, **params)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1, **params)
    elif model_name == "HistGradientBoosting":
        model = HistGradientBoostingClassifier(max_iter=150, random_state=42, early_stopping=False, **params)
    else:
        raise ValueError(model_name)
    return Pipeline([("preprocessor", make_preprocessor()), ("model", model)])


def ece_score(y_true, probability, bins: int = 10) -> tuple[float, list[dict]]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(y) != len(p) or len(y) == 0 or not np.isfinite(p).all():
        raise ValueError("ECE needs nonempty aligned finite probabilities")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignment = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    table = []
    total = len(y)
    value = 0.0
    for j in range(bins):
        take = assignment == j
        count = int(take.sum())
        mean_p = float(p[take].mean()) if count else None
        observed = float(y[take].mean()) if count else None
        if count:
            value += (count / total) * abs(observed - mean_p)
        table.append({"bin": j, "lower": float(edges[j]), "upper": float(edges[j + 1]), "count": count,
                      "mean_probability": mean_p, "observed_positive_rate": observed})
    return float(value), table


def metric_bundle(y_true, probability) -> tuple[dict, list[dict]]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("Metric inputs must be nonempty and aligned")
    if set(np.unique(y)) != {0, 1}:
        return ({"average_precision": None, "average_precision_undefined_reason": "evaluation data contain one class",
                 "roc_auc": None, "roc_auc_undefined_reason": "evaluation data contain one class",
                 "brier_score": float(brier_score_loss(y, p)), "ece": ece_score(y, p)[0]}, ece_score(y, p)[1])
    ece, bins = ece_score(y, p)
    return ({"average_precision": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
             "brier_score": float(brier_score_loss(y, p)), "ece": ece}, bins)


def confusion_metrics(y_true, flag, cost_ratio: float = 5.0) -> dict:
    y = np.asarray(y_true, dtype=int)
    f = np.asarray(flag, dtype=bool)
    tn, fp, fn, tp = confusion_matrix(y, f.astype(int), labels=[0, 1]).ravel().tolist()
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn), "precision": precision,
            "precision_undefined_reason": "no records flagged" if precision is None else None,
            "recall": recall, "recall_undefined_reason": "no positive labels" if recall is None else None,
            "false_positive_rate": fpr, "fpr_undefined_reason": "no negative labels" if fpr is None else None,
            "flagged_n": int(f.sum()), "flagged_fraction": float(f.mean()),
            "hypothetical_loss": float(fp + cost_ratio * fn)}


def threshold_candidates(probability: np.ndarray) -> np.ndarray:
    distinct = np.unique(np.asarray(probability, dtype=float))
    return np.concatenate([distinct, [np.nextafter(distinct[-1], np.inf)]])


def choose_threshold(y_true, probability, capacity: float, cost_ratio: float) -> tuple[float, dict]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    maximum_flags = math.floor(capacity * len(y) + 1e-12)
    candidates = []
    for threshold in threshold_candidates(p):
        flag = p >= threshold
        nflag = int(flag.sum())
        if nflag <= maximum_flags:
            metrics = confusion_metrics(y, flag, cost_ratio)
            candidates.append((metrics["hypothetical_loss"], nflag, -float(threshold), float(threshold), metrics))
    if not candidates:
        raise RuntimeError("The above-maximum no-selection threshold should always be feasible")
    _, _, _, best_threshold, best_metrics = min(candidates)
    return best_threshold, {**best_metrics, "capacity": float(capacity), "maximum_flags": maximum_flags,
                            "candidate_thresholds": int(len(threshold_candidates(p))),
                            "feasible_candidate_thresholds": int(len(candidates)), "threshold": float(best_threshold)}


def top_k_flags(probability, row_positions, capacity: float) -> tuple[np.ndarray, dict]:
    p = np.asarray(probability, dtype=float)
    rows = np.asarray(row_positions, dtype=int)
    k = math.floor(capacity * len(p) + 1e-12)
    order = np.lexsort((rows, -p))
    selected = order[:k]
    flag = np.zeros(len(p), dtype=bool)
    flag[selected] = True
    boundary = float(p[order[k - 1]]) if k else None
    ties = int(np.sum(p == boundary)) if k else 0
    return flag, {"capacity": float(capacity), "k": k, "boundary_probability": boundary,
                  "records_tied_at_boundary": ties, "achieved_fraction": float(flag.mean()),
                  "tie_rule": "descending calibrated probability, then ascending original zero-based row index"}


def generate_bootstrap_indices(n_samples: int, n_bootstrap: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_samples, size=(n_bootstrap, n_samples), dtype=np.int16)


def profile_keys(frame: pd.DataFrame) -> list[tuple]:
    return list(map(tuple, frame[FEATURES].to_numpy().tolist()))


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_experiment() -> dict:
    started = time.perf_counter()
    freeze, split_doc = verify_frozen_inputs()
    out = ROOT / "results"
    for folder in ["cv", "models", "predictions", "statistics", "calibration", "slices", "decisions", "figures", "diagnostics", "logs"]:
        (out / folder).mkdir(parents=True, exist_ok=True)
    frame = load_data()
    y_all = frame[TARGET].to_numpy(dtype=int)
    split = {k: np.asarray(v, dtype=int) for k, v in split_doc["indices_zero_based"].items()}
    train_rows = split["train_cv"]
    x_train = frame.loc[train_rows, FEATURES]
    y_train = y_all[train_rows]
    configs = configurations()
    fold_indices = split_doc["train_cv_folds"]
    cv_rows = []
    all_warnings = []
    selected = {}

    for model_name in MODEL_NAMES:
        model_start = time.perf_counter()
        summary = []
        print(f"Starting frozen CV: {model_name}; 12 configurations x 5 folds", flush=True)
        for config_index, params in enumerate(configs[model_name], start=1):
            fold_scores = []
            config_start = time.perf_counter()
            for fold_no, fold in enumerate(fold_indices, start=1):
                train_local = np.asarray(fold["train"], dtype=int)
                valid_local = np.asarray(fold["validation"], dtype=int)
                model = estimator(model_name, params)
                fit_start = time.perf_counter()
                status, ap, warning_text, failure = "ok", None, [], None
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        model.fit(x_train.iloc[train_local], y_train[train_local])
                    warning_text = [f"{w.category.__name__}: {w.message}" for w in caught]
                    prob = model.predict_proba(x_train.iloc[valid_local])[:, 1]
                    ap = float(average_precision_score(y_train[valid_local], prob))
                    fold_scores.append(ap)
                except Exception as exc:
                    status = "failed"
                    failure = f"{type(exc).__name__}: {exc}"
                elapsed = time.perf_counter() - fit_start
                convergence = [w for w in warning_text if "ConvergenceWarning" in w]
                all_warnings.extend([{"model": model_name, "config_index": config_index, "fold": fold_no, "warning": w} for w in warning_text])
                cv_rows.append({"model": model_name, "config_index": config_index, "params_json": json.dumps(params, sort_keys=True),
                                "fold": fold_no, "status": status, "average_precision": ap, "fit_seconds": elapsed,
                                "convergence_warning_count": len(convergence), "warnings_json": json.dumps(warning_text), "failure": failure})
            complete = len(fold_scores) == 5
            summary.append({"model": model_name, "config_index": config_index, "params": params, "complete": complete,
                            "mean_ap": float(np.mean(fold_scores)) if complete else None,
                            "std_ap": float(np.std(fold_scores, ddof=1)) if complete else None,
                            "config_fit_seconds": time.perf_counter() - config_start})
            print(f"  config {config_index:02d}/12 status={'complete' if complete else 'INCOMPLETE'} mean_AP={summary[-1]['mean_ap']}", flush=True)
        candidates = [r for r in summary if r["complete"]]
        if not candidates:
            write_csv(cv_rows, out / "cv/cv_fold_results.csv")
            raise RuntimeError(f"No complete configuration for {model_name}; stopped before calibration/test")
        best = sorted(candidates, key=lambda r: (-r["mean_ap"], r["config_index"]))[0]
        selected[model_name] = {**best, "model_cv_runtime_seconds": time.perf_counter() - model_start}
        print(f"Selected {model_name}: config {best['config_index']} mean AP={best['mean_ap']:.6f}", flush=True)
        write_csv([{**r, "params_json": json.dumps(r["params"], sort_keys=True), "params": None} for r in summary], out / f"cv/{model_name.lower().replace(' ', '_')}_configuration_summary.csv")
    write_csv(cv_rows, out / "cv/cv_fold_results.csv")
    (out / "logs/convergence_and_warnings.json").write_text(json.dumps(all_warnings, indent=2))
    (out / "cv/selected_models.json").write_text(json.dumps({k: {"config_index": v["config_index"], "params": v["params"], "mean_cv_ap": v["mean_ap"], "std_cv_ap": v["std_ap"]} for k, v in selected.items()}, indent=2))

    train_models = {}
    calibration_models = {}
    calibration_rows = []
    cal_idx = split["calibration"]
    threshold_idx = split["threshold_selection"]
    calibration_predictions = pd.DataFrame({"row_position": cal_idx, "ID": frame.iloc[cal_idx]["ID"].to_numpy(), "y_true": y_all[cal_idx]})
    threshold_predictions = pd.DataFrame({"row_position": threshold_idx, "ID": frame.iloc[threshold_idx]["ID"].to_numpy(), "y_true": y_all[threshold_idx]})

    for model_name in MODEL_NAMES:
        params = selected[model_name]["params"]
        base = estimator(model_name, params)
        fit_start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            base.fit(frame.iloc[train_rows][FEATURES], y_all[train_rows])
        refit_warnings = [f"{w.category.__name__}: {w.message}" for w in caught]
        probe_rows = train_rows[:min(64, len(train_rows))]
        base_before = base.predict_proba(frame.iloc[probe_rows][FEATURES])[:, 1].copy()
        frozen = FrozenEstimator(base)
        calibrator = CalibratedClassifierCV(estimator=frozen, method="sigmoid")
        calibrator.fit(frame.iloc[cal_idx][FEATURES], y_all[cal_idx])
        base_after = base.predict_proba(frame.iloc[probe_rows][FEATURES])[:, 1]
        unchanged = bool(np.array_equal(base_before, base_after))
        if not unchanged:
            raise RuntimeError(f"FrozenEstimator altered {model_name} base-model predictions")
        raw_cal = base.predict_proba(frame.iloc[cal_idx][FEATURES])[:, 1]
        calibrated_cal = calibrator.predict_proba(frame.iloc[cal_idx][FEATURES])[:, 1]
        raw_threshold = base.predict_proba(frame.iloc[threshold_idx][FEATURES])[:, 1]
        calibrated_threshold = calibrator.predict_proba(frame.iloc[threshold_idx][FEATURES])[:, 1]
        calibration_predictions[f"{model_name}__raw"] = raw_cal
        calibration_predictions[f"{model_name}__sigmoid"] = calibrated_cal
        threshold_predictions[f"{model_name}__raw"] = raw_threshold
        threshold_predictions[f"{model_name}__sigmoid"] = calibrated_threshold
        for score_type, scores in [("raw", raw_cal), ("sigmoid", calibrated_cal)]:
            metric, bins = metric_bundle(y_all[cal_idx], scores)
            calibration_rows.append({"model": model_name, "probability_type": score_type, **metric})
            for entry in bins:
                calibration_rows[-1].setdefault("bins", []).append(entry)
        model_key = model_name.lower().replace(" ", "_")
        with (out / f"models/{model_key}_base.pkl").open("wb") as f:
            pickle.dump(base, f, protocol=pickle.HIGHEST_PROTOCOL)
        with (out / f"models/{model_key}_sigmoid_calibrator.pkl").open("wb") as f:
            pickle.dump(calibrator, f, protocol=pickle.HIGHEST_PROTOCOL)
        train_models[model_name] = base
        calibration_models[model_name] = calibrator
        with (out / f"models/{model_key}_metadata.json").open("w") as f:
            json.dump({"params": params, "refit_train_rows": int(len(train_rows)), "calibration_rows": int(len(cal_idx)),
                       "calibrator": "CalibratedClassifierCV(FrozenEstimator(base), method='sigmoid')",
                       "base_predictions_unchanged_after_calibration_fit": unchanged,
                       "refit_fit_seconds": time.perf_counter() - fit_start, "refit_warnings": refit_warnings}, f, indent=2)
        print(f"Refit and independently calibrated {model_name}; frozen-base prediction equality={unchanged}", flush=True)
    calibration_predictions.to_csv(out / "predictions/calibration_predictions.csv", index=False)
    threshold_predictions.to_csv(out / "predictions/threshold_selection_predictions.csv", index=False)
    (out / "calibration/calibration_metrics.json").write_text(json.dumps(calibration_rows, indent=2))

    # Freeze all thresholds and capacity policies before generating any final-test score.
    decisions = {"status": "locked before final test predictions", "primary": {}, "sensitivity": [], "top_k": []}
    threshold_rows = []
    for model_name in MODEL_NAMES:
        p = threshold_predictions[f"{model_name}__sigmoid"].to_numpy()
        t = y_all[threshold_idx]
        th, summary = choose_threshold(t, p, 0.10, 5.0)
        decisions["primary"][model_name] = {"threshold": th, **summary}
        threshold_rows.append({"model": model_name, "scenario": "primary_10pct_fn5", **summary})
        for capacity in [0.05, 0.10, 0.20]:
            _, top_meta = top_k_flags(p, threshold_idx, capacity)
            decisions["top_k"].append({"model": model_name, **top_meta})
        for capacity, ratio in itertools.product([0.05, 0.10, 0.20], [2.0, 5.0, 10.0]):
            sens_th, sens_summary = choose_threshold(t, p, capacity, ratio)
            decisions["sensitivity"].append({"model": model_name, "capacity": capacity, "fn_to_fp_ratio": ratio,
                                             "threshold": sens_th, **sens_summary})
            threshold_rows.append({"model": model_name, "scenario": f"cap{capacity:.0%}_fn{ratio:g}", **sens_summary})
    (out / "decisions/locked_decisions.json").write_text(json.dumps(decisions, indent=2))
    write_csv(threshold_rows, out / "decisions/threshold_selection_records.csv")

    # Save Train-derived slice boundaries before any test labels or model scores are analyzed.
    train_frame = frame.iloc[train_rows]
    burden_ratio_train = train_frame["BILL_AMT1"].to_numpy(dtype=float) / train_frame["LIMIT_BAL"].to_numpy(dtype=float)
    burden_bounds = [float(v) for v in np.quantile(burden_ratio_train, [1/3, 2/3])]
    limit_bounds = [float(v) for v in np.quantile(train_frame["LIMIT_BAL"], [1/3, 2/3])]
    burden_slice_valid = bool(np.isfinite(burden_bounds).all() and burden_bounds[0] < burden_bounds[1])
    boundaries = {"derived_only_from_train_cv": True, "limit_balance_tertile_cutpoints": limit_bounds,
                  "signed_september_bill_to_limit_ratio": "BILL_AMT1 / LIMIT_BAL; negative values retained",
                  "bill_to_limit_tertile_cutpoints": burden_bounds,
                  "bill_to_limit_slice_status": "included" if burden_slice_valid else "omitted: training tertile cutpoints are not distinct and finite"}
    (out / "slices/train_derived_boundaries.json").write_text(json.dumps(boundaries, indent=2))

    # One-pass final-test scoring starts only after the locked decision file exists.
    test_idx = split["final_test"]
    test_frame = frame.iloc[test_idx]
    test_y = y_all[test_idx]
    test_predictions = pd.DataFrame({"row_position": test_idx, "ID": test_frame["ID"].to_numpy(), "y_true": test_y, "split": "final_test"})
    test_metrics = []
    reliability_rows = []
    curve_rows = []
    for model_name in MODEL_NAMES:
        base, calibrator = train_models[model_name], calibration_models[model_name]
        raw = base.predict_proba(test_frame[FEATURES])[:, 1]
        calibrated = calibrator.predict_proba(test_frame[FEATURES])[:, 1]
        test_predictions[f"{model_name}__raw"] = raw
        test_predictions[f"{model_name}__sigmoid"] = calibrated
        for score_type, prob in [("raw", raw), ("sigmoid", calibrated)]:
            metrics, bins = metric_bundle(test_y, prob)
            test_metrics.append({"model": model_name, "probability_type": score_type, **metrics})
            for b in bins:
                reliability_rows.append({"model": model_name, "probability_type": score_type, **b})
            precision, recall, _ = precision_recall_curve(test_y, prob)
            fpr, tpr, _ = roc_curve(test_y, prob)
            curve_rows.extend([{"model": model_name, "probability_type": score_type, "curve": "PR", "x": float(x), "y": float(yv)} for x, yv in zip(recall, precision)])
            curve_rows.extend([{"model": model_name, "probability_type": score_type, "curve": "ROC", "x": float(x), "y": float(yv)} for x, yv in zip(fpr, tpr)])
        dec = decisions["primary"][model_name]
        test_metrics.append({"model": model_name, "probability_type": "locked_threshold_test", **confusion_metrics(test_y, calibrated >= dec["threshold"], 5.0), "threshold": dec["threshold"]})
    test_predictions.to_csv(out / "predictions/final_test_predictions.csv", index=False)
    write_csv(test_metrics, out / "statistics/final_test_metrics.csv")
    write_csv(reliability_rows, out / "calibration/final_test_reliability_bins.csv")
    write_csv(curve_rows, out / "statistics/test_curve_points.csv")

    # Paired bootstrap: identical row draws for all models and score types.
    draws = generate_bootstrap_indices(len(test_idx))
    np.savez_compressed(out / "statistics/bootstrap_indices.npz", indices=draws, seed=np.array(BOOTSTRAP_SEED), test_row_positions=test_idx)
    replicate_rows = []
    one_class = 0
    for b, draw in enumerate(draws):
        yb = test_y[draw]
        two_classes = len(np.unique(yb)) == 2
        if not two_classes:
            one_class += 1
        for model_name in MODEL_NAMES:
            for score_type in ["raw", "sigmoid"]:
                p = test_predictions[f"{model_name}__{score_type}"].to_numpy()[draw]
                ece, _ = ece_score(yb, p)
                row = {"replicate": b, "model": model_name, "probability_type": score_type,
                       "average_precision": float(average_precision_score(yb, p)) if two_classes else None,
                       "roc_auc": float(roc_auc_score(yb, p)) if two_classes else None,
                       "brier_score": float(brier_score_loss(yb, p)), "ece": ece,
                       "ranking_metrics_valid": two_classes}
                replicate_rows.append(row)
    reps = pd.DataFrame(replicate_rows)
    reps.to_csv(out / "statistics/bootstrap_replicates.csv", index=False)
    interval_rows = []
    point_lookup = {(r["model"], r["probability_type"]): r for r in test_metrics if r["probability_type"] in ["raw", "sigmoid"]}
    for score_type in ["raw", "sigmoid"]:
        for metric in METRIC_NAMES:
            for model_name in MODEL_NAMES:
                vals = reps[(reps.model == model_name) & (reps.probability_type == score_type)][metric].dropna().to_numpy()
                base_row = reps[(reps.model == "Logistic Regression") & (reps.probability_type == score_type)][metric].to_numpy()
                model_row = reps[(reps.model == model_name) & (reps.probability_type == score_type)][metric].to_numpy()
                paired = model_row - base_row
                paired = paired[np.isfinite(paired)]
                point = point_lookup[(model_name, score_type)][metric]
                interval_rows.append({"model": model_name, "probability_type": score_type, "metric": metric,
                                      "test_estimate": point, "ci_95_low": float(np.percentile(vals, 2.5)),
                                      "ci_95_high": float(np.percentile(vals, 97.5)), "valid_replicates": int(len(vals)),
                                      "paired_difference_vs_logistic": float(point - point_lookup[("Logistic Regression", score_type)][metric]) if model_name != "Logistic Regression" else 0.0,
                                      "paired_difference_ci_95_low": float(np.percentile(paired, 2.5)),
                                      "paired_difference_ci_95_high": float(np.percentile(paired, 97.5)),
                                      "paired_valid_replicates": int(len(paired)),
                                      "one_class_resamples_rejected_for_ranking": one_class if metric in ["average_precision", "roc_auc"] else 0})
    write_csv(interval_rows, out / "statistics/bootstrap_intervals.csv")

    # Prespecified behavioral slices; all thresholds and boundaries were locked without test outcomes.
    pay0 = test_frame["PAY_0"].to_numpy()
    any_delay = test_frame[PAY].to_numpy().astype(float).max(axis=1) > 0
    limit_tier = pd.cut(test_frame["LIMIT_BAL"], [-np.inf, *limit_bounds, np.inf], labels=["low", "middle", "high"], include_lowest=True).astype(str).to_numpy()
    burden = test_frame["BILL_AMT1"].to_numpy(dtype=float) / test_frame["LIMIT_BAL"].to_numpy(dtype=float)
    burden_tier = (np.asarray(pd.cut(burden, [-np.inf, *burden_bounds, np.inf], labels=["low", "middle", "high"], include_lowest=True).astype(str))
                   if burden_slice_valid else None)
    pay0_group = np.where(pay0 <= -2, "-2", np.where(pay0 == -1, "-1", np.where(pay0 == 0, "0", ">=1")))
    slice_arrays = {"PAY_0 grouped": pay0_group, "Any PAY code > 0": np.where(any_delay, "true", "false"),
                    "LIMIT_BAL train tertile": limit_tier}
    if burden_slice_valid:
        slice_arrays["Signed bill-to-limit train tertile"] = burden_tier
    slice_rows = []
    for slice_name, labels in slice_arrays.items():
        for group in sorted(np.unique(labels)):
            mask = labels == group
            ys = test_y[mask]
            for model_name in MODEL_NAMES:
                prob = test_predictions.loc[mask, f"{model_name}__sigmoid"].to_numpy()
                metrics, _ = metric_bundle(ys, prob)
                dec = decisions["primary"][model_name]
                err = confusion_metrics(ys, prob >= dec["threshold"], 5.0)
                slice_rows.append({"slice": slice_name, "group": str(group), "model": model_name, "n": int(mask.sum()),
                                   "positive": int(ys.sum()), "negative": int(len(ys) - ys.sum()), **metrics,
                                   **{f"threshold_{k}": v for k, v in err.items()}})
    write_csv(slice_rows, out / "slices/final_test_behavioral_slices.csv")

    demographic_rows = []
    age = test_frame["AGE"].to_numpy()
    age_band = np.select([age < 30, age < 40, age < 50, age < 60], ["<30", "30-39", "40-49", "50-59"], default="60+")
    demographic_values = {"SEX": test_frame["SEX"].astype(str).to_numpy(), "EDUCATION": test_frame["EDUCATION"].astype(str).to_numpy(),
                          "MARRIAGE": test_frame["MARRIAGE"].astype(str).to_numpy(), "AGE_band": age_band}
    for field, labels in demographic_values.items():
        for group in sorted(np.unique(labels)):
            mask = labels == group
            ys = test_y[mask]
            for model_name in MODEL_NAMES:
                p = test_predictions.loc[mask, f"{model_name}__sigmoid"].to_numpy()
                th = decisions["primary"][model_name]["threshold"]
                err = confusion_metrics(ys, p >= th)
                demographic_rows.append({"field": field, "group": str(group), "model": model_name, "n": int(mask.sum()),
                                         "positive": int(ys.sum()), "negative": int(len(ys) - ys.sum()),
                                         "tp": err["tp"], "fp": err["fp"], "tn": err["tn"], "fn": err["fn"]})
    write_csv(demographic_rows, out / "slices/supplementary_demographic_error_audit.csv")

    # Exact 19-column feature-profile overlap; no scaling, encoding, or grouping is used.
    train_profiles = set(profile_keys(frame.iloc[train_rows]))
    overlap = np.asarray([key in train_profiles for key in profile_keys(test_frame)], dtype=bool)
    overlap_rows = []
    overlap_doc = {}
    for part in ["calibration", "threshold_selection", "final_test"]:
        idx = split[part]
        mask = np.asarray([key in train_profiles for key in profile_keys(frame.iloc[idx])], dtype=bool)
        overlap_doc[part] = {"n": int(len(mask)), "overlap_n": int(mask.sum()), "overlap_fraction": float(mask.mean()),
                             "nonoverlap_n": int((~mask).sum())}
    (out / "diagnostics/duplicate_profile_overlap.json").write_text(json.dumps(overlap_doc, indent=2))
    for group_name, mask in [("overlap", overlap), ("nonoverlap", ~overlap)]:
        ys = test_y[mask]
        for model_name in MODEL_NAMES:
            prob = test_predictions.loc[mask, f"{model_name}__sigmoid"].to_numpy()
            metrics, _ = metric_bundle(ys, prob)
            th = decisions["primary"][model_name]["threshold"]
            err = confusion_metrics(ys, prob >= th)
            overlap_rows.append({"profile_group": group_name, "model": model_name, "n": int(mask.sum()),
                                 "positive": int(ys.sum()), "negative": int(len(ys) - ys.sum()), **metrics,
                                 **{f"threshold_{k}": v for k, v in err.items()}})
    write_csv(overlap_rows, out / "diagnostics/test_overlap_group_metrics.csv")

    # Apply all locked sensitivity thresholds and Top-K policies to test, without retuning.
    sensitivity_test = []
    for record in decisions["sensitivity"]:
        model_name = record["model"]
        p = test_predictions[f"{model_name}__sigmoid"].to_numpy()
        metric = confusion_metrics(test_y, p >= record["threshold"], record["fn_to_fp_ratio"])
        sensitivity_test.append({**record, "test_threshold": record["threshold"], **{f"test_{k}": v for k, v in metric.items()}})
    write_csv(sensitivity_test, out / "decisions/cost_capacity_sensitivity.csv")
    top_rows = []
    for record in decisions["top_k"]:
        model_name = record["model"]
        p = test_predictions[f"{model_name}__sigmoid"].to_numpy()
        flags, meta = top_k_flags(p, test_idx, record["capacity"])
        top_rows.append({"model": model_name, **record, **{f"test_{k}": v for k, v in confusion_metrics(test_y, flags).items()},
                         "test_k": meta["k"], "test_boundary_probability": meta["boundary_probability"],
                         "test_records_tied_at_boundary": meta["records_tied_at_boundary"]})
    write_csv(top_rows, out / "decisions/top_k_test_results.csv")

    create_figures(test_predictions, test_y, test_metrics, reliability_rows, curve_rows, sensitivity_test, slice_rows, overlap_rows, out)
    summary = {"protocol_sha256": freeze["protocol_sha256"], "split_sha256": freeze["split_artifact_sha256"],
               "source_sha256": freeze["source_archive_sha256"], "python": sys.version, "platform": platform.platform(),
               "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__, "xlrd": xlrd.__version__,
               "bootstrap_replicates": BOOTSTRAP_N, "bootstrap_seed": BOOTSTRAP_SEED,
               "one_class_bootstrap_resamples": one_class, "wall_seconds": time.perf_counter() - started,
               "no_model_training_outside_three_frozen_candidates": True}
    (out / "experiment_manifest.json").write_text(json.dumps(summary, indent=2))
    print(f"Phase 2 run complete in {summary['wall_seconds']:.1f} seconds; one-class bootstrap draws={one_class}", flush=True)
    return summary


def create_figures(test_predictions, test_y, test_metrics, reliability_rows, curve_rows, sensitivity_test, slice_rows, overlap_rows, out: Path) -> None:
    colors = {"Logistic Regression": "#1f77b4", "Random Forest": "#ff7f0e", "HistGradientBoosting": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(8, 6))
    curve_frame = pd.DataFrame(curve_rows)
    for model in MODEL_NAMES:
        part = curve_frame[(curve_frame.model == model) & (curve_frame.curve == "PR") & (curve_frame.probability_type == "sigmoid")]
        ax.plot(part.x.to_numpy(), part.y.to_numpy(), color=colors[model], label=model)
    ax.set(xlabel="Recall", ylabel="Precision", title="Final Test Precision–Recall Curves (Sigmoid-Calibrated)")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out / "figures/pr_curves.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for model in MODEL_NAMES:
        part = curve_frame[(curve_frame.model == model) & (curve_frame.curve == "ROC") & (curve_frame.probability_type == "sigmoid")]
        ax.plot(part.x.to_numpy(), part.y.to_numpy(), color=colors[model], label=model)
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Final Test ROC Curves (Sigmoid-Calibrated)")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out / "figures/roc_curves.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for model in MODEL_NAMES:
        for kind, marker in [("raw", "o"), ("sigmoid", "s")]:
            rows = sorted([r for r in reliability_rows if r["model"] == model and r["probability_type"] == kind and r["count"] > 0], key=lambda r: r["bin"])
            ax.plot([r["mean_probability"] for r in rows], [r["observed_positive_rate"] for r in rows], marker=marker,
                    color=colors[model], linestyle="-" if kind == "sigmoid" else ":", label=f"{model} — {kind}")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive rate", title="Final Test Reliability Diagrams")
    ax.legend(fontsize=8); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out / "figures/reliability_diagrams.png", dpi=180); plt.close(fig)

    metric_frame = pd.DataFrame(test_metrics)
    metric_frame = metric_frame[metric_frame.probability_type.isin(["raw", "sigmoid"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in [(axes[0], "brier_score", "Brier Score (lower is better)"), (axes[1], "ece", "ECE (lower is better)")]:
        pivot = metric_frame.pivot(index="model", columns="probability_type", values=metric).reindex(MODEL_NAMES)
        pivot.plot(kind="bar", ax=ax, color={"raw": "#999999", "sigmoid": "#386cb0"})
        ax.set(title=title, xlabel="", ylabel=metric); ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Raw and Sigmoid-Calibrated Probability Quality")
    fig.tight_layout(); fig.savefig(out / "figures/calibration_comparison.png", dpi=180); plt.close(fig)

    sf = pd.DataFrame(sensitivity_test)
    sf["scenario"] = sf.apply(lambda r: f"{r['capacity']:.0%}\n{r['fn_to_fp_ratio']:g}:1", axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, model in zip(axes, MODEL_NAMES):
        part = sf[sf.model == model]
        for ratio, group in part.groupby("fn_to_fp_ratio"):
            group = group.sort_values("capacity")
            ax.plot(group.capacity * 100, group.test_hypothetical_loss, marker="o", label=f"FN:FP {ratio:g}:1")
        ax.set(title=model, xlabel="Threshold-selection capacity (%)")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Final Test hypothetical loss (units)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Locked Cost/Capacity Sensitivity on Final Test")
    fig.tight_layout(); fig.savefig(out / "figures/cost_capacity_sensitivity.png", dpi=180); plt.close(fig)

    sf_metric = pd.DataFrame(slice_rows)
    sf_metric = sf_metric[sf_metric.slice.isin(["PAY_0 grouped", "Any PAY code > 0", "LIMIT_BAL train tertile", "Signed bill-to-limit train tertile"])]
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_data = sf_metric.copy()
    plot_data["slice_group"] = plot_data["slice"] + "\n" + plot_data["group"]
    pivot = plot_data.pivot(index="slice_group", columns="model", values="average_precision").reindex(columns=MODEL_NAMES)
    pivot.plot(kind="bar", ax=ax, color=[colors[m] for m in MODEL_NAMES])
    ax.set(xlabel="Prespecified Final Test slice", ylabel="Average Precision", title="Final Test Slice Average Precision (Calibrated Scores)")
    ax.tick_params(axis="x", rotation=45); ax.grid(axis="y", alpha=.25); ax.legend(title="Model")
    fig.tight_layout(); fig.savefig(out / "figures/slice_average_precision.png", dpi=180); plt.close(fig)

    of = pd.DataFrame(overlap_rows)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    pivot = of.pivot(index="profile_group", columns="model", values="average_precision").reindex(columns=MODEL_NAMES)
    pivot.plot(kind="bar", ax=ax, color=[colors[m] for m in MODEL_NAMES])
    ax.set(xlabel="Exact 19-feature profile match to Train+CV", ylabel="Average Precision", title="Final Test Performance by Feature-Profile Overlap")
    ax.tick_params(axis="x", rotation=0); ax.grid(axis="y", alpha=.25); ax.legend(title="Model")
    fig.tight_layout(); fig.savefig(out / "figures/duplicate_profile_overlap.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    run_experiment()
