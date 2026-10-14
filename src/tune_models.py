
from pathlib import Path
import os

# Prevent numerical-library threading issues on macOS.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Project directories
# --------------------------------------------------

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed"
RESULTS = BASE / "results"

RESULTS.mkdir(parents=True, exist_ok=True)

EXCLUDED = {
    "match_id",
    "date",
    "team_1",
    "team_2",
    "team",
    "opponent",
    "goal_difference",
    "goals_scored",
}


# --------------------------------------------------
# 2. Model optimisation function
# --------------------------------------------------

def optimise_task(filename, target, task_name, expected_rows):

    df = pd.read_csv(DATA / filename)
    df["date"] = pd.to_datetime(df["date"])

    if len(df) != expected_rows:
        raise ValueError(
            f"{task_name}: expected {expected_rows} rows, "
            f"but found {len(df)}"
        )

    # Sort matches chronologically.
    matches = (
        df[["match_id", "date"]]
        .drop_duplicates()
        .sort_values(["date", "match_id"])
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 3. Chronological train/test split
    # --------------------------------------------------

    # Identify an approximately 80/20 split.
    split = int(len(matches) * 0.8)

    # Use the date of the first intended test match.
    cutoff_date = matches.iloc[split]["date"]

    # Keep all matches on the same date together.
    train_matches = matches[
        matches["date"] < cutoff_date
    ]

    test_matches = matches[
        matches["date"] >= cutoff_date
    ]

    train_ids = set(train_matches["match_id"])
    test_ids = set(test_matches["match_id"])

    train = df[df["match_id"].isin(train_ids)].copy()
    test = df[df["match_id"].isin(test_ids)].copy()

    # Check that training dates precede testing dates.
    assert train["date"].max() < test["date"].min()

    # Check that no match appears in both sets.
    assert train_ids.isdisjoint(test_ids)

    predictors = [
        column
        for column in df.columns
        if column not in EXCLUDED
    ]

    if len(predictors) != 8:
        raise ValueError(
            f"{task_name}: expected exactly 8 predictors, "
            f"found {len(predictors)}"
        )

    print(f"\n{'=' * 60}")
    print(task_name)
    print(f"{'=' * 60}")

    print(f"Total observations: {len(df)}")
    print(f"Training observations: {len(train)}")
    print(f"Testing observations: {len(test)}")
    print(f"Last training date: {train['date'].max().date()}")
    print(f"First testing date: {test['date'].min().date()}")
    print(f"Predictors: {predictors}")

    # --------------------------------------------------
    # 4. Define candidate regression models
    # --------------------------------------------------

    models = {
        "Baseline": DummyRegressor(strategy="mean"),

        "Linear Regression": make_pipeline(
            StandardScaler(),
            LinearRegression()
        ),
    }

    # Ridge regularisation strengths.
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        models[f"Ridge alpha={alpha}"] = make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha)
        )

    # Lasso regularisation strengths.
    for alpha in [0.001, 0.01, 0.1, 0.5, 1.0]:
        models[f"Lasso alpha={alpha}"] = make_pipeline(
            StandardScaler(),
            Lasso(
                alpha=alpha,
                max_iter=30000
            )
        )

    # --------------------------------------------------
    # 5. Time-aware cross-validation
    # --------------------------------------------------

    unique_dates = sorted(train["date"].unique())

    if len(unique_dates) < 6:
        raise ValueError(
            "Not enough training dates for cross-validation."
        )

    date_splitter = TimeSeriesSplit(n_splits=4)

    cv_rows = []

    for name, model in models.items():

        fold_mae = []

        for train_idx, val_idx in date_splitter.split(unique_dates):

            training_dates = set(
                unique_dates[i] for i in train_idx
            )

            validation_dates = set(
                unique_dates[i] for i in val_idx
            )

            fold_train = train[
                train["date"].isin(training_dates)
            ]

            fold_val = train[
                train["date"].isin(validation_dates)
            ]

            # Confirm strict chronological separation.
            assert (
                fold_train["date"].max()
                < fold_val["date"].min()
            )

            fitted = clone(model)

            fitted.fit(
                fold_train[predictors],
                fold_train[target]
            )

            predicted = fitted.predict(
                fold_val[predictors]
            )

            fold_mae.append(
                mean_absolute_error(
                    fold_val[target],
                    predicted
                )
            )

        cv_rows.append({
            "model": name,
            "cv_mae": float(np.mean(fold_mae)),
            "cv_std": float(np.std(fold_mae)),
        })

    cv_results = (
        pd.DataFrame(cv_rows)
        .sort_values("cv_mae")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 6. Select best model using CV only
    # --------------------------------------------------

    best_name = cv_results.iloc[0]["model"]

    best_model = clone(models[best_name])

    best_model.fit(
        train[predictors],
        train[target]
    )

    # Evaluate on the untouched testing period.
    test_pred = best_model.predict(
        test[predictors]
    )

    test_mae = mean_absolute_error(
        test[target],
        test_pred
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            test[target],
            test_pred
        )
    )

    test_r2 = r2_score(
        test[target],
        test_pred
    )

    # --------------------------------------------------
    # 7. Save results
    # --------------------------------------------------

    task_code = task_name.replace(
        " ", "_"
    ).replace(".", "_")

    cv_results.to_csv(
        RESULTS / f"{task_code}_cross_validation.csv",
        index=False
    )

    final_results = pd.DataFrame([{
        "task": task_name,
        "selected_model": best_name,
        "training_rows": len(train),
        "testing_rows": len(test),
        "last_training_date": str(
            train["date"].max().date()
        ),
        "first_testing_date": str(
            test["date"].min().date()
        ),
        "test_MAE": test_mae,
        "test_RMSE": test_rmse,
        "test_R2": test_r2,
    }])

    final_results.to_csv(
        RESULTS / f"{task_code}_optimised.csv",
        index=False
    )

    predictions = test[
        ["match_id", "date"]
    ].copy()

    if "team" in test.columns:
        predictions["team"] = test["team"].to_numpy()

    predictions["actual"] = test[target].to_numpy()
    predictions["predicted"] = test_pred
    predictions["residual"] = (
        predictions["actual"] - predictions["predicted"]
    )

    predictions.to_csv(
        RESULTS / f"{task_code}_optimised_predictions.csv",
        index=False
    )

    # --------------------------------------------------
    # 8. Display results
    # --------------------------------------------------

    print("\nTop five cross-validation models:")
    print(
        cv_results.head(5)
        .round(4)
        .to_string(index=False)
    )

    print(f"\nSelected model: {best_name}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R2: {test_r2:.4f}")

    print("\nResults saved successfully.")

    return final_results


# --------------------------------------------------
# 9. Run Regression 2.1
# --------------------------------------------------

results_21 = optimise_task(
    filename="regression_2_1.csv",
    target="goal_difference",
    task_name="Regression 2.1",
    expected_rows=104,
)


# --------------------------------------------------
# 10. Run Regression 2.2
# --------------------------------------------------

results_22 = optimise_task(
    filename="regression_2_2.csv",
    target="goals_scored",
    task_name="Regression 2.2",
    expected_rows=208,
)


# --------------------------------------------------
# 11. Completion message
# --------------------------------------------------

print("\nCross-validation and optimisation completed.")
print(f"Results saved in: {RESULTS}")
