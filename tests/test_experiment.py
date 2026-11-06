import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.experiment import (FEATURES, TARGET, choose_threshold, ece_score,
                                estimator, generate_bootstrap_indices,
                                load_data, make_preprocessor, metric_bundle,
                                top_k_flags, verify_frozen_inputs,
                                configurations, ROOT)


def test_frozen_hashes_dataset_and_split_integrity():
    freeze, splits = verify_frozen_inputs()
    assert len(freeze["protocol_sha256"]) == 64
    idx = splits["indices_zero_based"]
    combined = [position for values in idx.values() for position in values]
    assert len(combined) == 30000
    assert len(set(combined)) == 30000
    assert sorted(combined) == list(range(30000))
    for fold in splits["train_cv_folds"]:
        assert set(fold["train"]).isdisjoint(fold["validation"])
        assert len(fold["train"]) + len(fold["validation"]) == 18000


def test_source_schema_feature_allowlist_and_no_target_or_id_leakage():
    frame = load_data()
    assert TARGET == "default payment next month"
    assert len(FEATURES) == 19
    assert "ID" not in FEATURES and TARGET not in FEATURES
    assert set(frame.columns) == set(["ID", TARGET, "SEX", "EDUCATION", "MARRIAGE", "AGE"] + FEATURES)
    x = frame.loc[:250, FEATURES]
    transformer = make_preprocessor()
    transformed = transformer.fit_transform(x)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(x)
    assert transformed.shape[1] > len(FEATURES)
    assert not any("ID" in name or TARGET in name for name in transformer.get_feature_names_out())


def test_all_candidates_share_identical_preprocessor_information():
    names = ["Logistic Regression", "Random Forest", "HistGradientBoosting"]
    pipes = [estimator(name, {}) for name in names]
    assert all(pipe.named_steps["preprocessor"].get_params(deep=True).keys() == pipes[0].named_steps["preprocessor"].get_params(deep=True).keys() for pipe in pipes)
    assert all(pipe.named_steps["preprocessor"].transformers[0][2] == ["LIMIT_BAL"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)] for pipe in pipes)
    assert all(pipe.named_steps["preprocessor"].transformers[1][2] == ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"] for pipe in pipes)


def test_ece_equal_width_definition_and_one_class_metric_handling():
    value, table = ece_score([0, 1], [0.05, 0.95])
    assert np.isclose(value, 0.05)
    assert len(table) == 10 and sum(row["count"] for row in table) == 2
    assert table[-1]["count"] == 1
    metrics, _ = metric_bundle([1, 1], [0.2, 0.8])
    assert metrics["average_precision"] is None
    assert metrics["roc_auc"] is None
    assert metrics["average_precision_undefined_reason"]


def test_threshold_capacity_and_deterministic_top_k_ties():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6])
    threshold, record = choose_threshold(y, p, capacity=0.5, cost_ratio=1)
    assert record["flagged_n"] <= 2
    assert np.sum(p >= threshold) == record["flagged_n"]
    tied_flags, meta = top_k_flags([0.8, 0.8, 0.8], [10, 3, 5], capacity=1/3)
    assert tied_flags.tolist() == [False, True, False]
    assert meta["records_tied_at_boundary"] == 3


def test_paired_bootstrap_is_deterministic_and_shared():
    a = generate_bootstrap_indices(50, n_bootstrap=20, seed=2026)
    b = generate_bootstrap_indices(50, n_bootstrap=20, seed=2026)
    assert a.shape == (20, 50)
    assert np.array_equal(a, b)
    assert np.all((a >= 0) & (a < 50))


def test_saved_paired_bootstrap_recomputes_same_draw_for_each_model():
    test = pd.read_csv(ROOT / "results/predictions/final_test_predictions.csv")
    replicates = pd.read_csv(ROOT / "results/statistics/bootstrap_replicates.csv")
    with np.load(ROOT / "results/statistics/bootstrap_indices.npz") as saved:
        draws = saved["indices"]
        assert int(saved["seed"]) == 2026
        assert draws.shape == (2000, len(test))
        for replicate in [0, 119, 1999]:
            draw = draws[replicate].astype(int)
            yb = test.y_true.to_numpy(dtype=int)[draw]
            assert set(np.unique(yb)) == {0, 1}
            for model in ["Logistic Regression", "Random Forest", "HistGradientBoosting"]:
                pb = test[f"{model}__raw"].to_numpy()[draw]
                row = replicates[(replicates.replicate == replicate) &
                                 (replicates.model == model) &
                                 (replicates.probability_type == "raw")].iloc[0]
                assert np.isclose(row.average_precision, average_precision_score(yb, pb))
                assert np.isclose(row.roc_auc, roc_auc_score(yb, pb))
                assert np.isclose(row.brier_score, brier_score_loss(yb, pb))


def test_sigmoid_calibration_uses_frozen_base_and_preserves_base_predictions():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(100, 3))
    y = (x[:, 0] + 0.2 * x[:, 1] > 0).astype(int)
    base = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
    base.fit(x[:60], y[:60])
    before = base.predict_proba(x[:10]).copy()
    calibration = CalibratedClassifierCV(estimator=FrozenEstimator(base), method="sigmoid")
    calibration.fit(x[60:80], y[60:80])
    after = base.predict_proba(x[:10])
    assert np.array_equal(before, after)
    assert calibration.predict_proba(x[80:]).shape == (20, 2)
    assert isinstance(calibration.estimator, FrozenEstimator)


def test_final_prediction_row_alignment_contract():
    frame = load_data()
    _, splits = verify_frozen_inputs()
    test_idx = np.asarray(splits["indices_zero_based"]["final_test"])
    aligned = frame.iloc[test_idx]
    assert np.array_equal(aligned["ID"].to_numpy(), np.asarray(splits["row_ids"]["final_test"]))
    assert len(aligned[TARGET].to_numpy()) == len(test_idx)


def test_frozen_grids_and_locked_thresholds_match_saved_test_application():
    assert {name: len(items) for name, items in configurations().items()} == {
        "Logistic Regression": 12, "Random Forest": 12, "HistGradientBoosting": 12}
    decisions = __import__("json").loads((ROOT / "results/decisions/locked_decisions.json").read_text())
    test = pd.read_csv(ROOT / "results/predictions/final_test_predictions.csv")
    saved = pd.read_csv(ROOT / "results/statistics/final_test_metrics.csv")
    for model, record in decisions["primary"].items():
        threshold = record["threshold"]
        flags = test[f"{model}__sigmoid"].to_numpy() >= threshold
        row = saved[(saved.model == model) & (saved.probability_type == "locked_threshold_test")].iloc[0]
        assert np.isclose(row.threshold, threshold)
        assert int(row.flagged_n) == int(flags.sum())
        assert int(row.tp + row.fp) == int(flags.sum())
    topk = pd.read_csv(ROOT / "results/decisions/top_k_test_results.csv")
    assert len(topk) == 9
    assert not topk.duplicated(["model", "capacity"]).any()
