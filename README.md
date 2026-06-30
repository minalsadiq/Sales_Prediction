# Sales Prediction Using Python

## Overview
This project forecasts product sales based on advertising spend
(TV, radio, social media), pricing, target customer segment, and
advertising platform, helping quantify how marketing decisions drive
revenue outcomes.

## Objective
Build and compare **Linear Regression** and **Random Forest**
regression models to predict sales, analyze advertising-channel impact,
and deliver actionable marketing insights.

# 📊 Dataset Description

The dataset contains sales-related information used to build prediction models.

Example features:

| Feature | Description |
|----------|-------------|
| Date | Sales time period |
| Product | Product information |
| Region | Sales location |
| Segment | Customer category |
| Platform | Sales source |
| Quantity | Units sold |
| Revenue | Sales amount |


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


  # 📈 Sales Prediction using Machine Learning

## Overview

Sales Prediction is a Machine Learning project developed to forecast future sales by analyzing historical sales data and identifying patterns that influence business performance.

The goal of this project is to transform raw business data into meaningful predictions that can support planning and decision-making.

This project follows a complete machine learning workflow including:

- Data Collection
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Model Development
- Model Evaluation
- Visualization
- Prediction Analysis

Multiple regression models were implemented and compared to understand their prediction capabilities.

---

# 📌 Problem Statement

Businesses generate large amounts of sales data every day.

Understanding future sales trends manually is difficult and time-consuming.

This project aims to predict sales values using historical data and machine learning techniques while identifying important factors affecting sales performance.

---

# 🎯 Objectives

The objectives of this project are:

✔ Analyze historical sales data  
✔ Discover sales patterns and trends  
✔ Build predictive machine learning models  
✔ Compare model performance  
✔ Generate visual insights  
✔ Improve understanding of business forecasting  

---


### Target Variable

Sales Prediction

---

# 🧹 Data Preprocessing

Data preprocessing was performed before training.

### 1. Data Loading
Imported dataset into Python.

### 2. Data Cleaning
Handled missing values and removed inconsistencies.

### 3. Feature Engineering
Prepared useful variables for training.

### 4. Data Transformation
Converted required variables into machine-readable format.

### 5. Train-Test Split
Separated training and testing datasets.

---

# 📈 Exploratory Data Analysis (EDA)

Several visual analyses were created to understand sales behaviour.

Generated Visualizations:

### Sales Trend Analysis
Shows how sales change over time.

### Sales Distribution
Understanding sales spread.

### Correlation Heatmap
Identifies relationships among variables.

### Sales by Segment & Platform
Compares performance across categories.

### Feature Importance
Shows variables contributing most to predictions.

### Prediction Comparison
Visual comparison between actual and predicted values.

---

# 🤖 Machine Learning Models

Two regression models were implemented.

---

## 1. Linear Regression

Baseline prediction model.

Features:

- Simple implementation
- Fast execution
- Easy interpretation

---

## 2. Random Forest Regression

Ensemble-based prediction model.

Features:

- Captures complex relationships
- Strong prediction capability
- Better generalization

---

# 📉 Model Evaluation

Model performance was evaluated using multiple metrics.

Evaluation Metrics:

- R² Score
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

Generated Results:

```plaintext
Linear Regression Report

Random Forest Report

Prediction Comparison Graph

Model Comparison Graph
```

---

# 📊 Results & Findings

Key observations from this project:

- Historical sales patterns were successfully analyzed.
- Multiple models were trained and evaluated.
- Visual outputs improved result interpretation.
- Feature importance identified major business drivers.
- Prediction comparison showed model performance differences.

This project demonstrated how machine learning can support business forecasting and decision-making.
<img width="1536" height="1024" alt="Dashboard" src="https://github.com/user-attachments/assets/b3da7b67-9418-4897-bc4f-afba37d741e2" />

---


# 🛠 Technologies Used

| Category | Tools |
|----------|-------|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Machine Learning | Scikit-learn |
| Development | VS Code |

---

# 🚀 Installation

Clone repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

Move into folder:

```bash
cd Task4_Sales_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

# 🔮 Future Improvements

Possible enhancements:

- Hyperparameter Optimization
- Advanced Forecasting Models
- Interactive Dashboard
- Deployment with Streamlit
- Real-Time Sales Prediction

---

# 👩‍💻 Author

**Minal**  
Data Science Student  
Machine Learning Enthusiast

---

⭐ If you found this project useful, consider giving it a star.
