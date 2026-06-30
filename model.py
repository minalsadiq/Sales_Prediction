"""
model.py
Regression model training logic for the Sales Prediction project.
Trains and compares Linear Regression and Random Forest Regressor, and
builds a forecast comparison summary.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

import config
from utils import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Trains, compares, and persists sales-forecasting regression models."""

    def __init__(self):
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=300, random_state=config.RANDOM_STATE
            ),
        }
        self.trained_models = {}
        self.cv_scores = {}

    def train_all(self, X_train, y_train):
        """Fit every candidate model and record 5-fold CV R^2."""
        for name, model in self.models.items():
            logger.info("Training %s ...", name)
            model.fit(X_train, y_train)
            self.trained_models[name] = model

            scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
            self.cv_scores[name] = scores.mean()
            logger.info("%s -> CV R^2: %.4f", name, scores.mean())
        return self.trained_models

    def get_best_model(self):
        """Select the model with the highest cross-validated R^2."""
        best_name = max(self.cv_scores, key=self.cv_scores.get)
        best_model = self.trained_models[best_name]
        logger.info(
            "Best model selected: %s (CV R^2 = %.4f)", best_name, self.cv_scores[best_name]
        )
        return best_name, best_model

    @staticmethod
    def build_forecast_comparison(results: dict, y_test) -> pd.DataFrame:
        """
        Build a side-by-side comparison of actual vs. each model's
        predicted sales values for the test set.
        """
        comparison = pd.DataFrame({"actual_sales": y_test.values})
        for name, preds in results.items():
            comparison[f"predicted_{name.replace(' ', '_').lower()}"] = preds
        return comparison.reset_index(drop=True)

    @staticmethod
    def advertising_impact_analysis(model, preprocessor, X_train_columns) -> dict:
        """
        Analyze the impact of advertising spend on sales using the
        Linear Regression coefficients (if applicable), mapping each
        coefficient back to its original feature name.
        """
        if not hasattr(model, "coef_"):
            return {}
        feature_names = preprocessor.get_feature_names_out()
        impact = dict(zip(feature_names, model.coef_))
        return impact

    @staticmethod
    def save_model(model, path: str = config.BEST_MODEL_PATH):
        joblib.dump(model, path)
        logger.info("Model saved to %s", path)

    @staticmethod
    def save_preprocessor(preprocessor, path: str = config.PREPROCESSOR_PATH):
        joblib.dump(preprocessor, path)
        logger.info("Preprocessor saved to %s", path)
