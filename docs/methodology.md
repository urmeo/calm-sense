# Methodology and recorded analyses

Saved experiments: 4 October 2026. [Primary scores](../README.md#recorded-results) · [Run commands](reproduction.md) · [Machine-readable results](../outputs/results/)

## Data and windows

The primary benchmark uses WESAD chest recordings at 700 Hz from 15 subjects: S2 through S17, excluding S12. Eligible labels are baseline, stress, and amusement. Meditation, undefined labels, and ignored segments are excluded. Binary classification uses baseline and stress; three-class classification also includes amusement.

The pipeline filters signals and computes 60-second windows every 30 seconds. A window is retained only when at least 90% of its labels belong to the same eligible class. The recorded dataset contains 869 binary windows and 1,032 three-class windows. Chest channels are ECG, EDA, temperature, respiration, and acceleration magnitude. The CNN receives these processed channels resampled to 1,024 samples per window.

Sixty-second windows limit interpretation of very-low-frequency HRV and long-term detrended fluctuation analysis.

The feature registry contains 60 measurements:

| Signal group | Registered features |
| :-- | --: |
| Heart rate variability (HRV) | 30 |
| Electrodermal activity (EDA) | 15 |
| Temperature and respiration | 10 |
| Acceleration | 5 |

The recorded benchmark drops `RESP_inhale_exhale_ratio` and `RESP_variability`, which are entirely missing, leaving 58 features. Remaining missing values are median-imputed within each training fold, followed by training-fold standardization. Feature definitions are implemented in [src/features/](../src/features/).

## Models and evaluation

The primary experiment uses 15 leave-one-subject-out folds. Each fold holds out one complete subject. Accuracy and macro-F1 are averaged over subject scores; macro-F1 uses the full task class set. Pooled accuracy and balanced accuracy combine held-out predictions across subjects. Binary AUROC and AUPRC are pooled metrics for the four feature models; they were not computed for the CNN.

These measurements cover 15 laboratory participants. They do not establish performance in daily life or clinical settings. Live inference latency and wearable battery use have not been measured.

| Model | Main settings |
| :-- | :-- |
| Logistic Regression | L2 penalty, `C=1`, balanced class weights |
| Random Forest | 200 trees, maximum depth 10, minimum leaf size 5, balanced class weights |
| XGBoost | 200 trees, maximum depth 7, learning rate 0.1, balanced training sample weights |
| LightGBM | 200 trees, 50 leaves, learning rate 0.1, balanced class weights |
| 1D-CNN | Residual convolution blocks, AdamW, early stopping |

The CNN separates training and validation subjects inside each outer fold. Normalization statistics use only the inner training windows. Tuning is a separate binary experiment with outer LOSO evaluation and inner grouped folds. The exported Random Forest uses the fixed defaults, not the tuning results. See [classifier defaults](../src/models/ml/classifiers.py) and [CNN training](../src/models/dl/cnn_1d.py) for the full settings.

The Random Forest mean binary accuracy is 91.0%; its pooled balanced accuracy is 89.9%. Its subject accuracies range from 71.2% to 100.0%. A bootstrap over subjects with 10,000 repetitions gives a 95% interval of 85.8% to 95.6% for mean accuracy. The Friedman test across the four feature models gives `p = 0.599`. No pairwise Wilcoxon comparison is significant after Holm correction. These tests do not establish that one feature model is superior. Full results are in [stats.json](../outputs/results/stats.json).

## Subject-mixed comparison

The comparison uses the same non-overlapping subset for LOSO and stratified five-fold testing. The five-fold split mixes subjects between training and test sets. These pooled scores show how the evaluation split changes the result:

| Task and model | LOSO accuracy | Subject-mixed accuracy | Difference |
| :-- | --: | --: | --: |
| Binary, Random Forest | 91.3% | 96.4% | 5.0 percentage points |
| Three-class, LightGBM | 67.2% | 94.0% | 26.8 percentage points |

The differences use unrounded values. Subject-mixed testing does not measure generalization to an unseen subject. The full-window LOSO means in the README use a different aggregation and window set, so they should not be substituted into this comparison.

## Calibration and personalization

Calibration uses binary Random Forest predictions. Each outer fold fits its calibrator to grouped out-of-fold predictions from training subjects. The held-out subject is excluded from global calibration. The saved analysis reports 15-bin expected calibration error (ECE), maximum calibration error (MCE), and the Brier score. ECE and MCE compare the predicted class confidence with bin accuracy. Binary Brier is the mean squared error of the stress probability.

Pooled ECE on all LOSO windows changes from 0.077 to 0.026 with isotonic recalibration. MCE changes from 0.177 to 0.350, so the mean improvement does not extend to the worst bin. Brier changes from 0.069 to 0.064. The matched non-overlapping analysis is reported separately. The [README tables](../README.md#calibration-and-personalization) are generated from [calibration.json](../outputs/results/calibration.json) and [personalization.json](../outputs/results/personalization.json).

Personalization first retains every other held-out window to remove overlap. It reserves a stratified half for evaluation and draws class-balanced labeled enrollment windows from the other half. Only the probability calibrator changes; the classifier is not retrained. Scores are means across subjects on this reserved half, so they are not directly comparable with pooled full-window calibration scores.

Requested budgets are 5, 10, and 20 windows. A budget of 5 selects 4 balanced windows because the sampler takes the same integer count from both classes. Limited class availability can reduce other budgets. With a requested budget of 20, mean ECE changes from 0.162 without calibration to 0.078; the training-subject global calibrator gives 0.099. These results are exploratory and do not establish an enrollment procedure for live use. [Decision curves](../outputs/figures/calibration_decision_curve.png) are also an offline analysis, without a validated clinical decision threshold.

Ablation, calibration, and personalization analyses have no multiplicity correction.

## Ablation, wrist recordings, and feature analysis

The binary ablation experiment reruns Random Forest LOSO evaluation for each feature subset:

| Subset | Features | Mean accuracy | Mean macro-F1 |
| :-- | --: | --: | --: |
| All features | 58 | 0.910 | 0.895 |
| Without motion | 53 | 0.901 | 0.891 |
| HRV and EDA | 45 | 0.894 | 0.883 |
| HRV only | 30 | 0.810 | 0.777 |
| EDA only | 15 | 0.828 | 0.803 |
| Motion only | 5 | 0.885 | 0.870 |

The high motion-only score is a reason to examine protocol effects before interpreting these features as specific measures of stress. The saved [ablation results](../outputs/results/ablation.csv) do not establish causation or performance outside the laboratory protocol.

The wrist experiment uses WESAD wrist BVP, EDA, temperature, and acceleration at their native sampling rates. A comparison using Random Forest on both devices gives chest accuracy 0.910 and wrist accuracy 0.890. Wrist XGBoost has mean accuracy 0.901. These are separate feature pipelines and sensor sets; the same-model comparison holds the classifier choice fixed. See [wrist.json](../outputs/results/wrist.json).

SHAP values describe a separate XGBoost model fitted on all binary feature windows. They are descriptive feature contributions for that fit, not held-out estimates or causal effects. [shap_top_features.csv](../outputs/results/shap_top_features.csv) records the mean absolute contributions.

## Cross-dataset transfer

Transfer uses 18 shared features from WESAD wrist and PhysioNet Non-EEG recordings. The shared signals are EDA, temperature, acceleration magnitude, and heart rate. Windows are 60 seconds with 50% overlap. The recorded WESAD subset has 15 subjects and 869 windows: 562 baseline and 307 stress. Non-EEG has 20 subjects and 1,133 windows: 709 relaxation and 424 stress.

All transfer comparisons use Random Forest. Within-dataset macro-F1 is a subject mean; balanced accuracy is pooled across held-out LOSO predictions. Cross-dataset scores pool predictions over the complete target dataset.

For Non-EEG, cognitive and emotional stress map to stress, relaxation maps to baseline, and physical stress and other annotations are excluded. These labels are not equivalent to WESAD's laboratory conditions. Feature slopes use seconds, and heart rate samples are assigned by timestamps within each window.

| Signal | WESAD units | Non-EEG units |
| :-- | :-- | :-- |
| EDA | Microsiemens | Normalized units |
| Temperature | Degrees Celsius | Degrees Celsius |
| Acceleration | 1/64 g | Normalized units |
| Heart rate | Beats per minute | Beats per minute |

The experiment retains the EDA and acceleration unit differences. It does not establish sensor equivalence.

| Evaluation | Balanced accuracy | Macro-F1 |
| :-- | --: | --: |
| Within WESAD, LOSO | 0.868 | 0.852 |
| Within Non-EEG, LOSO | 0.701 | 0.673 |
| Train WESAD, test Non-EEG | 0.558 | 0.544 |
| Train Non-EEG, test WESAD | 0.497 | 0.495 |

The poor transfer scores limit claims of generalization across devices or populations. This is one dataset pair, with differences in units, devices, labels, and stressors. Full counts, unit metadata, and provenance are in [cross_dataset.json](../outputs/results/cross_dataset.json).

## Figures and provenance

The [plot index](../outputs/README.md#plot-index) links every saved chart to its measurements. Confusion matrices pool default classifier decisions and normalize rows.

The [provenance record](../outputs/results/provenance.json) identifies source revision `4f5c02f`, seed `42`, feature schema and benchmark protocol version `2`, and portable schema version `3`. It records package versions, input verification, and artifact hashes. Saved results remain measurements of that source revision. New source changes need new measurements.
