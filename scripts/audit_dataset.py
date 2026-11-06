#!/usr/bin/env python3
"""Deterministic, read-only audit for the original UCI XLS workbook.

This script inspects and reports; it does not modify the source or fit models.
Run: python scripts/audit_dataset.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xlrd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/raw/uci_default_credit_card_350.zip"
OUT = ROOT / "results/audit"
TARGET = "default payment next month"
PREDICTORS = ["LIMIT_BAL", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]
DEMOGRAPHIC = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]
EXPECTED = ["ID"] + ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)] + [TARGET]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with zipfile.ZipFile(ARCHIVE) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".xls")]
        if len(members) != 1:
            raise RuntimeError(f"Expected one original .xls, found {members}")
        member = members[0]
        # Header is the second row in UCI's original workbook; explicitly skip metadata row.
        with zf.open(member) as stream:
            frame = pd.read_excel(stream, header=1, engine="xlrd")
    frame.columns = [str(c).strip() for c in frame.columns]
    if TARGET not in frame:
        raise ValueError(f"Target {TARGET!r} absent; columns={frame.columns.tolist()}")
    y = pd.to_numeric(frame[TARGET], errors="coerce")
    freq = {c: {str(k): int(v) for k, v in frame[c].value_counts(dropna=False).sort_index().items()} for c in ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"] if c in frame}
    numeric = frame.select_dtypes(include=[np.number])
    feature = frame[[c for c in PREDICTORS if c in frame]]
    all_features = frame[[c for c in frame.columns if c not in {"ID", TARGET}]]
    feature_key = feature.astype("string").fillna("<NA>").agg("|".join, axis=1)
    all_feature_key = all_features.astype("string").fillna("<NA>").agg("|".join, axis=1)

    def duplicate_diagnostics(keys: pd.Series) -> dict:
        sizes = keys.value_counts(dropna=False)
        labels = pd.DataFrame({"key": keys, "y": y}).groupby("key", dropna=False).y
        dup_sizes = sizes[sizes > 1]
        return {
            "duplicate_vector_groups": int(len(dup_sizes)),
            "rows_in_duplicate_vector_groups": int(dup_sizes.sum()),
            "groups_with_conflicting_labels": int((labels.nunique(dropna=False) > 1).sum()),
        }

    X = y.to_numpy()
    if y.isna().any() or not set(y.dropna().unique()).issubset({0, 1}):
        raise ValueError("Cannot stratify: target has missing or nonbinary values")
    all_idx = np.arange(len(frame))
    train_cv, remainder = train_test_split(all_idx, test_size=.40, random_state=42, stratify=y)
    calib, remainder = train_test_split(remainder, test_size=.625, random_state=42, stratify=y.iloc[remainder])
    threshold, test = train_test_split(remainder, test_size=.60, random_state=42, stratify=y.iloc[remainder])
    splits = {"train_cv": train_cv, "calibration": calib, "threshold_selection": threshold, "final_test": test}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = [{"train": tr.tolist(), "validation": va.tolist()} for tr, va in cv.split(train_cv, y.iloc[train_cv])]
    fold_support = [{
        "fold": i + 1,
        "train_positive": int(y.iloc[train_cv[fold["train"]]].sum()),
        "train_negative": int(len(fold["train"]) - y.iloc[train_cv[fold["train"]]].sum()),
        "validation_positive": int(y.iloc[train_cv[fold["validation"]]].sum()),
        "validation_negative": int(len(fold["validation"]) - y.iloc[train_cv[fold["validation"]]].sum()),
    } for i, fold in enumerate(folds)]
    if any(min(v["train_positive"], v["train_negative"], v["validation_positive"], v["validation_negative"]) == 0 for v in fold_support):
        raise AssertionError("A CV fold lacks a target class")
    joined = np.concatenate(list(splits.values()))
    if len(joined) != len(frame) or len(np.unique(joined)) != len(frame):
        raise AssertionError("Split coverage/disjointness validation failed")
    row_ids = {name: frame.iloc[idx]["ID"].astype("int64").tolist() for name, idx in splits.items()}
    split_doc = {"seed": 42, "status": "provisional; requires human approval", "indices_zero_based": {k: v.tolist() for k, v in splits.items()}, "row_ids": row_ids, "train_cv_folds_indices_are_local_to_train_cv_order": True, "train_cv_folds": folds, "fold_class_support": fold_support}
    (OUT / "proposed_splits.json").write_text(json.dumps(split_doc, indent=2))
    split_summary = {k: {"n": int(len(v)), "positive": int(y.iloc[v].sum()), "negative": int(len(v) - y.iloc[v].sum()), "prevalence": float(y.iloc[v].mean())} for k, v in splits.items()}
    slices = {}
    limit_cutpoints = [float(v) for v in frame.iloc[train_cv]["LIMIT_BAL"].quantile([1/3, 2/3]).tolist()]
    for name, idx in splits.items():
        block = frame.iloc[idx]
        block_y = y.iloc[idx]
        delinquent = block[[f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]]].gt(0).any(axis=1)
        tiers = pd.cut(block["LIMIT_BAL"], bins=[-np.inf, *limit_cutpoints, np.inf], labels=["low", "middle", "high"], include_lowest=True)

        def labeled_counts(values: pd.Series) -> dict:
            tab = pd.crosstab(values, block_y, dropna=False)
            return {str(k): {"n": int(row.sum()), "negative": int(row.get(0, 0)), "positive": int(row.get(1, 0))} for k, row in tab.iterrows()}

        slices[name] = {
            "PAY_0": {str(k): int(v) for k, v in block["PAY_0"].value_counts(dropna=False).sort_index().items()},
            "PAY_0_label_support": labeled_counts(block["PAY_0"]),
            "any_PAY_positive": {"true": int(delinquent.sum()), "false": int((~delinquent).sum())},
            "any_PAY_positive_label_support": labeled_counts(delinquent),
            "LIMIT_BAL_train_tertiles": {str(k): int(v) for k, v in tiers.value_counts(sort=False, dropna=False).items()},
            "LIMIT_BAL_train_tertiles_label_support": labeled_counts(tiers),
        }
    audit = {
        "provenance": {"dataset": "UCI Default of Credit Card Clients, ID 350", "source_page": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients", "download_url": "https://archive.ics.uci.edu/static/public/350/default%2Bof%2Bcredit%2Bcard%2Bclients.zip", "retrieved_local_date": "2026-11-06 (Asia/Singapore)", "archive": ARCHIVE.relative_to(ROOT).as_posix(), "archive_bytes": ARCHIVE.stat().st_size, "archive_sha256": sha256(ARCHIVE), "original_filename": member, "original_bytes": zf.getinfo(member).file_size, "license": "CC BY 4.0; cite Yeh (2009), DOI 10.24432/C55S3H"},
        "environment": {"python": sys.version, "platform": platform.platform(), "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__, "xlrd": xlrd.__version__},
        "data": {"rows": int(len(frame)), "columns": int(frame.shape[1]), "column_names": frame.columns.tolist(), "dtypes": {c: str(t) for c, t in frame.dtypes.items()}, "expected_schema_match": frame.columns.tolist() == EXPECTED, "PAY_0_present": "PAY_0" in frame, "PAY_1_present": "PAY_1" in frame, "target": TARGET, "target_binary_valid": bool(not y.isna().any() and set(y.unique()).issubset({0, 1})), "target_counts": {str(k): int(v) for k, v in y.value_counts(dropna=False).sort_index().items()}, "target_prevalence": float(y.mean()), "missing_by_column": {c: int(v) for c, v in frame.isna().sum().items()}, "nonfinite_by_column": {c: int((~np.isfinite(numeric[c])).sum()) for c in numeric}, "ranges": {c: {"min": float(numeric[c].min()), "max": float(numeric[c].max())} for c in numeric}, "categorical_frequencies": freq, "category_review": {"EDUCATION_codes_outside_documented_1_to_4": sorted(int(v) for v in set(frame.EDUCATION.unique()) - {1, 2, 3, 4}), "MARRIAGE_codes_outside_documented_1_to_3": sorted(int(v) for v in set(frame.MARRIAGE.unique()) - {1, 2, 3}), "PAY_codes_requiring_semantic_review": sorted(int(v) for v in set(np.concatenate([frame[c].unique() for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]])) - set([-1, *range(1, 10)]))}},
        "duplicates": {"missing_ids": int(frame.ID.isna().sum()), "duplicate_ids": int(frame.ID.duplicated(keep=False).sum()), "exact_duplicate_rows": int(frame.duplicated(keep=False).sum()), "all_predictors_excluding_ID_target": duplicate_diagnostics(all_feature_key), "primary_19_predictors": duplicate_diagnostics(feature_key)},
        "feature_policy": {"primary_allowlist": PREDICTORS, "target_excluded": TARGET not in PREDICTORS, "ID_excluded": "ID" not in PREDICTORS, "demographics_retained_only_for_supplementary_audit": DEMOGRAPHIC},
        "splits": split_summary, "audit_slices": slices, "train_cv_limit_balance_tertile_cutpoints": limit_cutpoints, "cv_folds": len(folds), "cv_fold_class_support": fold_support, "split_indices_file": "proposed_splits.json", "audit_runtime_seconds": time.perf_counter() - started,
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False))
    print(json.dumps({k: audit[k] for k in ["provenance", "environment", "data", "duplicates", "feature_policy", "splits", "cv_folds", "audit_runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
