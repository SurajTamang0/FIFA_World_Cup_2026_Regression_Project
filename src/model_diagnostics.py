
from pathlib import Path
import os

# Limit numerical library threads on macOS.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
FIGURES = BASE / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def find_column(df, candidates):
    """Find a column using common naming conventions."""
    lookup = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def create_diagnostics(task, target):
    prefix = f"Regression_{task}"
    prediction_file = RESULTS / f"{prefix}_optimised_predictions.csv"
    score_file = RESULTS / f"{prefix}_scores.csv"

    print(f"\n{'=' * 55}")
    print(f"Regression {task}")
    print(f"{'=' * 55}")

    if not prediction_file.exists():
        print(f"Missing prediction file: {prediction_file.name}")
        return

    df = pd.read_csv(prediction_file)
    print("Prediction columns:", df.columns.tolist())

    actual_col = find_column(
        df,
        ["actual", "y_true", "y_test", "true", target]
    )
    predicted_col = find_column(
        df,
        ["predicted", "prediction", "y_pred", "predicted_value",
         "predicted_goals", "predicted_goal_difference"]
    )

    if actual_col is None or predicted_col is None:
        print("Could not identify actual and predicted columns.")
        print("Please send me a screenshot of the output.")
        return

    actual = pd.to_numeric(df[actual_col], errors="coerce")
    predicted = pd.to_numeric(df[predicted_col], errors="coerce")

    valid = actual.notna() & predicted.notna()
    actual = actual[valid].to_numpy()
    predicted = predicted[valid].to_numpy()

    if len(actual) == 0:
        print("No valid prediction rows were found.")
        return

    residuals = actual - predicted

    # Figure 1: Actual versus predicted
    plt.figure(figsize=(7, 5))
    plt.scatter(actual, predicted, alpha=0.75)

    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())
    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        label="Perfect prediction"
    )

    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    plt.title(f"Regression {task}: Actual vs Predicted")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        FIGURES / f"{prefix}_actual_vs_predicted.png",
        dpi=300
    )
    plt.close()

    # Figure 2: Residual distribution
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=12, edgecolor="black")
    plt.axvline(0, linestyle="--")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title(f"Regression {task}: Residual Distribution")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        FIGURES / f"{prefix}_residual_distribution.png",
        dpi=300
    )
    plt.close()

    # Figure 3: Residuals versus predicted values
    plt.figure(figsize=(7, 5))
    plt.scatter(predicted, residuals, alpha=0.75)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted value")
    plt.ylabel("Residual")
    plt.title(f"Regression {task}: Residual Diagnostics")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        FIGURES / f"{prefix}_residuals_vs_predicted.png",
        dpi=300
    )
    plt.close()

    # Figure 4: Model comparison, when available
    if score_file.exists():
        scores = pd.read_csv(score_file)
        model_col = find_column(scores, ["model", "model_name"])
        mae_col = find_column(scores, ["MAE", "test_mae"])

        if model_col is not None and mae_col is not None:
            scores = scores.dropna(subset=[model_col, mae_col])
            scores[mae_col] = pd.to_numeric(
                scores[mae_col], errors="coerce"
            )
            scores = scores.dropna(subset=[mae_col])

            if not scores.empty:
                plt.figure(figsize=(8, 5))
                plt.barh(
                    scores[model_col].astype(str),
                    scores[mae_col]
                )
                plt.xlabel("Test Mean Absolute Error (lower is better)")
                plt.title(f"Regression {task}: Model Comparison")
                plt.tight_layout()
                plt.savefig(
                    FIGURES / f"{prefix}_model_comparison.png",
                    dpi=300
                )
                plt.close()

    print(f"Valid test predictions: {len(actual)}")
    print(f"Mean residual: {np.mean(residuals):.4f}")
    print(f"Mean absolute error: {np.mean(np.abs(residuals)):.4f}")
    print("Diagnostic graphs generated.")


create_diagnostics("2_1", "goal_difference")
create_diagnostics("2_2", "goals_scored")

print("\nFinished. Check the figures folder.")
