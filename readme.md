# EV Range Anxiety and Adoption Prediction

A machine learning pipeline that predicts whether a person is likely to buy an electric vehicle (`will_buy_ev`) from demographics, commute distance, charging access, range anxiety, and related behavioral features.

The main workflow lives in a Jupyter notebook. Shared helpers for plotting, outlier filtering, and model training sit in `src/`.

## Highlights

- End-to-end notebook: EDA, preprocessing, training, and evaluation
- Four model pipelines: Logistic Regression, Random Forest, K-Nearest Neighbors, and XGBoost
- Stratified train/test split, class-imbalance handling, and hyperparameter search (GridSearchCV / RandomizedSearchCV)
- Evaluation with accuracy, precision, recall, F1, ROC AUC, and confusion matrices
- Optional Intel GPU offload for sklearn-compatible estimators (off by default; toggle in the notebook)

## Research question

How do **range anxiety level** and **daily commute distance**—together with charging access and other user attributes—relate to the likelihood of EV adoption?

## Project structure

```
ml_pipeline.ipynb      End-to-end analysis and modeling notebook
src/
  data_utils.py        Distribution plots and z-score outlier filtering
  ml_utils.py          Pipeline construction, training wrapper, evaluation
data/                  Local dataset directory (not committed; see Data below)
requirements.txt       Pinned Python dependencies
changelog.md           Project history
license                MIT License
```

## Requirements

- Python 3.11+ recommended (matches the pinned scientific stack)
- Dependencies listed in `requirements.txt` (scikit-learn, XGBoost, pandas, numpy, scipy, matplotlib, seaborn, Jupyter, Intel Extension for Scikit-learn)

## Data

The notebook loads:

```text
data/ev_adoption.csv
```

The dataset is **proprietary**, so `data/` and CSV files are gitignored and are not in the public repository. Put your local copy at `data/ev_adoption.csv` before running the notebook.

Expected columns include `age`, `gender`, `annual_income_usd`, `city_type`, `daily_commute_km`, charging-related fields, `environmental_concern_level`, `range_anxiety_level`, and the target `will_buy_ev`.

## GPU acceleration

Training uses CPU by default (`USE_GPU = False` in `src/ml_utils.py`). In the first code cell of `ml_pipeline.ipynb`, call `set_gpu_enabled(True)` to try Intel GPU offload via `scikit-learn-intelex`, or leave `set_gpu_enabled(False)` for CPU. If GPU offload fails, training falls back to CPU automatically. This mainly affects sklearn estimators; XGBoost is not accelerated the same way.

## Installation

```bash
git clone https://github.com/yoav-es/ml_pipeline_for_ev_range_anxiety.git
cd ml_pipeline_for_ev_range_anxiety
```

If you cloned the local folder name instead:

```bash
cd ml_pipeline_for_ev_range_anxiety_prediciton
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
```

Run the notebook from the project root so the `data/` path and `src/` imports resolve:

```bash
jupyter notebook ml_pipeline.ipynb
```

## Pipeline overview

1. **EDA** — Normalize column names, inspect missing values, filter z-score outliers on selected numeric columns, plot distributions.
2. **Features** — Predictors such as income, city type, commute distance, charging access, range anxiety, and environmental concern; target is `will_buy_ev` (Yes/No).
3. **Preprocessing** — `ColumnTransformer` with `SimpleImputer` (mean for numeric, most frequent for categorical), one-hot encoding, and `StandardScaler`.
4. **Models** — Logistic Regression with RFE, Random Forest, KNN with RFE, XGBoost with RFE; 5-fold CV scoring on F1.
5. **Results** — Logistic Regression is the strongest overall trade-off on the reported run (highest F1 and ROC AUC). KNN accuracy is inflated by the ~83% / ~17% class imbalance.

## License

MIT. See `license`.
