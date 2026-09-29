# Changelog

## 07/09/26

- Initial repository creation and baseline project setup.

## 19/09/26

- Refined the ETL stage: histograms and z-score outlier analysis.
- Converted ETL toward an ML-oriented format and added a pipeline template.

## 22/09/26

- Added a Random Forest benchmark pipeline with RandomizedSearchCV and RFE.
- Included `environmental_concern_level`.
- Split numerical and categorical preprocessing explicitly.
- Added a stratified train-test split.
- Added out-of-fold cross-validation for classification threshold optimization, plus evaluation metrics and confusion matrices.

## 25/09/26

- Added and tested K-Nearest Neighbors and XGBoost pipelines.
- Documented GPU integration issues and findings.

## 26/09/26

- Refactored analysis results and conclusions.

## 28/09/26

- Added a GPU execution wrapper for conditional hardware selection during training.
- Moved core pipeline helpers into modular backend functions under `src/`.
- Rechecked evaluation results, metrics, and summary tables.
- Refactored analysis insights and final conclusions.

## 29/09/26

- Rewrote the README to match the actual project layout, data path, and setup steps.
- Fixed `requirements.txt` (correct Jupyter package name; added packages imported by `src/ml_utils.py`).
- Deduplicated and cleaned this changelog.
