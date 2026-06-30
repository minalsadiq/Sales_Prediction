"""
data_preprocessing.py
Handles dataset acquisition, cleaning, feature engineering, and
preprocessing for the Sales Prediction project.
"""

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import config
from utils import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Encapsulates all data loading, cleaning, and preprocessing logic."""

    def __init__(self, dataset_path: str = config.DATASET_PATH):
        self.dataset_path = dataset_path
        self.preprocessor = self._build_preprocessor()

    # ------------------------------------------------------------------
    # Dataset acquisition
    # ------------------------------------------------------------------
    def load_or_create_dataset(self) -> pd.DataFrame:
        """
        Load the advertising/sales dataset from disk, generating a
        realistic synthetic dataset on first run if no local copy exists.
        """
        if os.path.exists(self.dataset_path):
            logger.info("Loading existing dataset from %s", self.dataset_path)
            df = pd.read_csv(self.dataset_path)
        else:
            logger.info("Dataset not found locally. Generating synthetic dataset.")
            df = self._generate_synthetic_dataset()
            df.to_csv(self.dataset_path, index=False)
            logger.info("Dataset saved to %s", self.dataset_path)
        return df

    @staticmethod
    def _generate_synthetic_dataset(n_samples: int = 800) -> pd.DataFrame:
        """Generate a realistic synthetic advertising-spend/sales dataset."""
        rng = np.random.default_rng(config.RANDOM_STATE)

        segments = ["Youth", "Adults", "Seniors", "Families"]
        platforms = ["TV", "Social Media", "Radio", "Mixed"]
        segment_multiplier = {"Youth": 1.15, "Adults": 1.0, "Seniors": 0.85, "Families": 1.05}
        platform_multiplier = {"TV": 1.05, "Social Media": 1.2, "Radio": 0.9, "Mixed": 1.1}

        records = []
        for _ in range(n_samples):
            tv_spend = float(rng.uniform(0, 300))
            radio_spend = float(rng.uniform(0, 50))
            social_spend = float(rng.uniform(0, 100))
            price = float(rng.uniform(10, 100))
            segment = rng.choice(segments)
            platform = rng.choice(platforms)

            base_sales = (
                4.0
                + 0.045 * tv_spend
                + 0.18 * radio_spend
                + 0.09 * social_spend
                - 0.03 * price
            )
            base_sales *= segment_multiplier[segment] * platform_multiplier[platform]
            noise = rng.normal(0, 1.2)
            sales = max(base_sales + noise, 0.5)

            records.append(
                {
                    "tv_spend": round(tv_spend, 2),
                    "radio_spend": round(radio_spend, 2),
                    "social_spend": round(social_spend, 2),
                    "price": round(price, 2),
                    "target_segment": segment,
                    "platform": platform,
                    config.TARGET_COLUMN: round(sales, 2),
                }
            )

        df = pd.DataFrame(records)
        missing_idx = rng.choice(df.index, size=int(len(df) * 0.015), replace=False)
        df.loc[missing_idx, "social_spend"] = np.nan
        return df

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicates and impute missing numeric values with the median."""
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info("Removed %d duplicate rows", before - len(df))

        for col in config.NUMERIC_FEATURES:
            if df[col].isnull().sum() > 0:
                n_missing = df[col].isnull().sum()
                df[col] = df[col].fillna(df[col].median())
                logger.info("Imputed %d missing values in '%s' with median", n_missing, col)
        return df

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add total advertising spend and channel-mix ratio features."""
        df = df.copy()
        df["total_ad_spend"] = df["tv_spend"] + df["radio_spend"] + df["social_spend"]
        for extra in ("total_ad_spend",):
            if extra not in config.NUMERIC_FEATURES:
                config.NUMERIC_FEATURES.append(extra)
                config.FEATURE_COLUMNS.append(extra)
        return df

    # ------------------------------------------------------------------
    # Preprocessing pipeline
    # ------------------------------------------------------------------
    @staticmethod
    def _build_preprocessor() -> ColumnTransformer:
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        categorical_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, config.NUMERIC_FEATURES),
                ("cat", categorical_transformer, config.CATEGORICAL_FEATURES),
            ]
        )

    def split_and_transform(self, df: pd.DataFrame):
        X = df[config.FEATURE_COLUMNS]
        y = df[config.TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )

        X_train_t = self.preprocessor.fit_transform(X_train)
        X_test_t = self.preprocessor.transform(X_test)

        logger.info("Split data: %d train rows, %d test rows", len(X_train), len(X_test))
        return X_train_t, X_test_t, y_train, y_test

    def run_pipeline(self):
        df = self.load_or_create_dataset()
        df = self.clean_data(df)
        df = self.engineer_features(df)
        self.preprocessor = self._build_preprocessor()
        X_train, X_test, y_train, y_test = self.split_and_transform(df)
        return df, X_train, X_test, y_train, y_test
