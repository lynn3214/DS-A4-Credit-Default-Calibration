"""Focused unit tests for the independent Phase 2 verifier."""
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_phase2_independently.py"
SPEC = importlib.util.spec_from_file_location("independent_verifier", MODULE_PATH)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def aligned_predictions():
    return pd.DataFrame({"row_position": [2, 5], "ID": [12, 15], "y_true": [0, 1],
                         "split": ["final_test", "final_test"]})


def test_alignment_detects_row_order_and_label_mismatch():
    frame = aligned_predictions()
    verifier.validate_prediction_alignment(frame, [2, 5], [12, 15], [0, 1])
    with pytest.raises(ValueError, match="row positions"):
        verifier.validate_prediction_alignment(frame.iloc[::-1], [2, 5], [12, 15], [0, 1])
    with pytest.raises(ValueError, match="labels"):
        verifier.validate_prediction_alignment(frame, [2, 5], [12, 15], [1, 0])


def test_alignment_detects_duplicate_ids():
    frame = aligned_predictions()
    frame.loc[1, "ID"] = 12
    with pytest.raises(ValueError, match="Duplicate customer ID"):
        verifier.validate_prediction_alignment(frame, [2, 5], [12, 15], [0, 1])


@pytest.mark.parametrize("values", [[-0.01, 0.5], [0.2, 1.01], [0.2, np.nan]])
def test_probability_validation_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        verifier.require_probability(values)


def test_ece_uses_left_closed_bins_and_includes_one_in_last_bin():
    value, bins = verifier.independent_ece([0, 0, 1, 1], [0.0, 0.1, 0.999, 1.0])
    assert bins[0]["count"] == 1
    assert bins[1]["count"] == 1
    assert bins[9]["count"] == 2
    assert bins[9]["mean_probability"] == pytest.approx(0.9995)
    assert value == pytest.approx((0.0 + 0.1 + abs(0.9995 - 1.0) * 2) / 4)


def test_confusion_and_cost_are_independently_calculated():
    result = verifier.independent_confusion([1, 1, 0, 0], [1, 0, 1, 0], fn_cost=5, fp_cost=1)
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (1, 1, 1, 1)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["false_positive_rate"] == pytest.approx(0.5)
    assert result["hypothetical_loss"] == 6


def test_critical_slice_tp_expectations_distinguish_the_two_PAY_groups():
    assert verifier.EXPECTED_SLICE_TPS["PAY_0_equals_zero"] == {
        "Logistic Regression": 0, "Random Forest": 1, "HistGradientBoosting": 0}
    assert verifier.EXPECTED_SLICE_TPS["all_six_PAY_nonpositive"] == {
        "Logistic Regression": 0, "Random Forest": 0, "HistGradientBoosting": 0}


def test_bootstrap_comparison_rejects_nonpaired_replicate_values():
    labels = np.array([0, 1, 0, 1])
    scores = {"Logistic Regression__raw": np.array([0.1, 0.9, 0.2, 0.8]),
              "Random Forest__raw": np.array([0.2, 0.8, 0.1, 0.9])}
    paired_indices = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int16)
    recomputed = verifier.recompute_bootstrap_rows(labels, scores, paired_indices)
    saved = pd.DataFrame(recomputed)
    saved[["model", "probability_type"]] = saved.model_probability.str.rsplit("__", n=1, expand=True)
    saved = saved.drop(columns="model_probability")
    verifier.compare_bootstrap_replicates(recomputed, saved)
    saved.loc[(saved.model == "Random Forest") & (saved.replicate == 1), "average_precision"] += 0.05
    with pytest.raises(ValueError, match="not paired/reproducible"):
        verifier.compare_bootstrap_replicates(recomputed, saved)


def test_verifier_does_not_import_experimental_calculation_helpers():
    source = MODULE_PATH.read_text()
    assert "from scripts.experiment" not in source
    assert "import scripts.experiment" not in source
