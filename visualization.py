"""
visualization.py
Visualizations for the Sales Prediction project: sales trend, feature
importance, and prediction comparison plots. All plots auto-saved to
results/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config
from utils import get_logger

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")


class Visualizer:
    """Generates and saves all charts for the project."""

    def __init__(self, results_dir: str = config.RESULTS_DIR):
        self.results_dir = results_dir

    def _save(self, fig, filename: str):
        path = os.path.join(self.results_dir, filename)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info("Saved figure: %s", path)

    def plot_sales_vs_spend(self, df):
        """Scatter plots of sales vs each advertising spend channel."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        channels = ["tv_spend", "radio_spend", "social_spend"]
        for ax, channel in zip(axes, channels):
            sns.regplot(
                data=df, x=channel, y=config.TARGET_COLUMN, ax=ax,
                scatter_kws={"alpha": 0.4}, line_kws={"color": "red"},
            )
            ax.set_title(f"Sales vs {channel.replace('_', ' ').title()}")
        fig.suptitle("Advertising Spend Impact on Sales")
        fig.tight_layout()
        self._save(fig, "sales_trend.png")

    def plot_correlation_heatmap(self, df):
        fig, ax = plt.subplots(figsize=(7, 6))
        corr = df[config.NUMERIC_FEATURES + [config.TARGET_COLUMN]].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        self._save(fig, "correlation_heatmap.png")

    def plot_sales_by_segment_platform(self, df):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.boxplot(data=df, x="target_segment", y=config.TARGET_COLUMN, ax=axes[0])
        axes[0].set_title("Sales by Target Segment")
        sns.boxplot(data=df, x="platform", y=config.TARGET_COLUMN, ax=axes[1])
        axes[1].set_title("Sales by Platform")
        fig.tight_layout()
        self._save(fig, "sales_by_segment_platform.png")

    def plot_feature_importance(self, model, feature_names, model_name: str):
        if not hasattr(model, "feature_importances_"):
            logger.info("%s has no feature_importances_ attribute; skipping.", model_name)
            return
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:15]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], ax=ax, palette="magma")
        ax.set_title(f"Feature Importance - {model_name}")
        self._save(fig, f"feature_importance_{model_name.replace(' ', '_').lower()}.png")

    def plot_prediction_comparison(self, comparison_df, n=50):
        """Line chart comparing actual vs predicted sales for a sample of test points."""
        subset = comparison_df.head(n).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(subset.index, subset["actual_sales"], label="Actual", marker="o", color="black")
        for col in subset.columns:
            if col.startswith("predicted_"):
                label = col.replace("predicted_", "").replace("_", " ").title()
                ax.plot(subset.index, subset[col], label=label, marker="x", linestyle="--")
        ax.set_title("Actual vs Predicted Sales (Sample of Test Set)")
        ax.set_xlabel("Test Sample Index")
        ax.set_ylabel("Sales")
        ax.legend()
        self._save(fig, "prediction_comparison.png")

    def plot_model_comparison(self, comparison_df):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.barplot(data=comparison_df, x="Model", y="R2", ax=axes[0], palette="crest")
        axes[0].set_title("Model R^2 Comparison")
        sns.barplot(data=comparison_df, x="Model", y="RMSE", ax=axes[1], palette="flare")
        axes[1].set_title("Model RMSE Comparison")
        fig.tight_layout()
        self._save(fig, "model_comparison.png")
