# EV Adoption Prediction Pipeline

A machine learning pipeline to analyze and predict electric vehicle (EV) adoption based on user demographics, behavioral, and vehicle characteristics. The project delivers a reproducible workflow for identifying key predictors associated with willingness to buy an EV.

## Quick Highlights

- **Notebook**: `notebooks/ev_adoption_analysis.ipynb` — End-to-end data analysis, modeling, and results
- **Package**: `ev_pipeline/` — Modular helpers for data loading, preprocessing, feature engineering, and modeling
- **Environment**: Python 3.10+ with core scientific and ML libraries
- **Execution**: Supports local CLI and interactive Jupyter workflows

## Overview

- The goal of this project is to analyze and predict EV adoption by evaluating user demographics, daily commute distances, and range anxiety levels through a robust machine learning pipeline.
   By examining behavioral patterns, coverage distributions, and metric alignments, the model identifies key predictive features that influence the likelihood of a person buying an EV. Ultimately, these insights provide actionable strategies to inform vehicle manufacturing, infrastructure development, and policy decisions.


## Features

- End-to-end ML pipeline with modular preprocessing and feature engineering
- Multiple model pipelines (Logistic Regression, Random Forest, KNN, XGBoost) with hyperparameter tuning
- Automated feature selection and threshold optimization
- Robust evaluation metrics and confusion matrix reporting

## Project Structure

- `notebooks/ev_adoption_analysis.ipynb` — Primary interactive analysis notebook
- `ev_pipeline/` — Python source package containing modular helpers (data loading, processing, analysis, visualization)
- `tests/` — Unit and integration tests
- `scripts/` — Helper and automation scripts
- `requirements.txt` / `pyproject.toml` — Dependency and build setup files

## Requirements

- Python 3.10+
- numpy
- pandas
- scikit-learn
- scipy
- xgboost
- matplotlib
- seaborn
- jupyter-notebook

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone [repository_url]
   cd [repository_directory]
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Windows PowerShell: .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   pip install -e .             # Optional: install package in editable mode
   ```

## Running the Notebook

Launch the interactive Jupyter notebook environment:

```bash
jupyter notebook ml_pipeline
```
