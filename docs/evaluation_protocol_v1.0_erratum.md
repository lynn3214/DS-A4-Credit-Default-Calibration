# Evaluation Protocol v1.0 Erratum

**Issued:** 2026-11-08, during independent verification of the saved Phase 2 results.

## Paired-difference direction

The frozen protocol's Section 8 states: “Negative differences favor the compared model for AP/AUC; positive differences favor it for Brier/ECE.” That sentence reverses the sign interpretation for all four metrics.

The saved experimental convention is `delta = compared_model_metric - Logistic_Regression_metric`. Therefore, a positive AP or ROC-AUC difference favors the compared model, while a negative Brier or ECE difference favors it. This follows from the direction in which each metric is better: larger for AP/AUC and smaller for Brier/ECE.

The independent verification recomputed replicate-wise differences from the preserved shared bootstrap indices and confirmed the saved numerical differences use the compared-model-minus-Logistic-Regression convention. The mistake is explanatory wording only. It does not affect code, saved estimates, confidence intervals, thresholds, or decision rules. The frozen protocol is retained unchanged; its SHA-256 remains `03ffa36a02fcdbbf8df8484ffdb84f58e063819899409812cb4e46a87abae418`, matching the freeze manifest.

This erratum records a correction discovered after the protocol was frozen. It does not change or retroactively revise the pretraining protocol.
