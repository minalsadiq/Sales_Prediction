# Task 4 — Sales Prediction Using Python

## Overview
This project forecasts product sales based on advertising spend
(TV, radio, social media), pricing, target customer segment, and
advertising platform, helping quantify how marketing decisions drive
revenue outcomes.

## Objective
Build and compare **Linear Regression** and **Random Forest**
regression models to predict sales, analyze advertising-channel impact,
and deliver actionable marketing insights.

## Project Structure
```
Task4_Sales_Prediction/
├── main.py                 # Orchestrates the full pipeline
├── data_preprocessing.py   # Loading, cleaning, feature engineering, encoding
├── model.py                  # Model training, model selection, impact analysis
├── evaluation.py             # MAE, RMSE, R^2 evaluation
├── visualization.py          # Sales trend, importance, prediction comparison plots
├── config.py                  # Paths and hyperparameters
├── utils.py                    # Logging helpers
├── requirements.txt
├── .gitignore
├── datasets/                  # Dataset + instructions
├── models/                     # Saved model + preprocessor (created at runtime)
└── results/                    # Generated plots and reports (created at runtime)
```

## How to Run
```bash
pip install -r requirements.txt
python main.py
```

## Methodology
1. **Data Preprocessing** — load (or synthetically generate) the
   advertising/sales dataset, drop duplicates, impute missing numeric
   values with the median.
2. **Feature Engineering** — derive `total_ad_spend`; numeric features
   are standardized and categorical features (`target_segment`,
   `platform`) are one-hot encoded via a `ColumnTransformer`.
3. **Model Training** — Linear Regression and Random Forest are
   trained and evaluated with 5-fold cross-validation (R²).
4. **Evaluation** — MAE, RMSE, and R² are computed on a held-out test
   set for every model.
5. **Advertising Impact Analysis** — Linear Regression coefficients are
   mapped back to feature names to quantify how each advertising
   channel and segment affects sales.
6. **Forecast Comparison** — a side-by-side table/plot of actual vs.
   each model's predicted sales values.
7. **Model Selection** — the model with the highest cross-validated R²
   is selected and persisted to `models/best_model.joblib`.

## Results
All generated artifacts are saved automatically to `results/`:
- `sales_trend.png` (sales vs. each advertising channel)
- `correlation_heatmap.png`, `sales_by_segment_platform.png`
- `feature_importance_<model>.png` for tree-based models
- `prediction_comparison.png` (actual vs predicted overlay)
- `model_comparison.png` / `model_comparison.csv`
- Per-model text reports (`*_report.txt`)

## Key Learnings
- TV and social media spend show the strongest positive relationship
  with sales; radio has a smaller but still positive effect.
- Random Forest captures non-linear interactions between channels and
  segments that linear regression misses, typically improving R².
- The advertising-impact coefficients give marketers a quantified,
  actionable lever for budget allocation decisions.
