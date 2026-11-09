# SPAIQ Results Summary Draft

Can nonlinear learners improve next-month default ranking and probability quality over a logistic baseline, and how do their scores behave under a capacity-limited review rule? We evaluated the UCI Default of Credit Card Clients dataset (30,000 records; historical Taiwan cohort) under frozen stratified partitions: 18,000 Train+CV, 4,500 calibration, 3,000 threshold selection, and 4,500 Final Test. Logistic Regression, Random Forest, and HistGradientBoosting shared the same folds and test cases; model-specific sigmoid calibrators and thresholds were locked before Final Test evaluation.

Final Test AP was 0.5574, 0.5750, and 0.5798, respectively; HGB ROC-AUC was 0.7882. Sigmoid calibration did not uniformly improve Brier score or ten-bin ECE: both worsened for LR and RF and improved slightly for HGB. Under the hypothetical FP=1/FN=5 policy capped at 10% on threshold-selection data, Final Test losses were 3,603 (LR), 3,528 (RF), and 3,541 (HGB). The 13-unit RF advantage over HGB is a point difference without a direct paired uncertainty interval.

HGB is the leading candidate for further offline ranking evaluation, while RF minimizes the observed frozen-policy loss; neither result establishes statistical superiority across models. This random split represents one 2005 Taiwan cohort. Intramonth feature availability, temporal generalization, contemporary-population performance, and intervention effects remain unverified. The results support further validation, not operational lending use.

**Word count:** 221 (summary text only; regex word-token count). The 150–300 range is a provisional target from the local task specification; it was not confirmed on the public course pages or current SPAIQ task page.
