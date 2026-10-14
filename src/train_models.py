
from pathlib import Path
import os

# Limit numerical-library threads to avoid the import issue on this Mac.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "processed"
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def evaluate_task(filename, target, task_name, expected_rows):
    df = pd.read_csv(DATA / filename)
    df["date"] = pd.to_datetime(df["date"])

    if len(df) != expected_rows:
        raise ValueError(
            f"{task_name}: expected {expected_rows} rows, found {len(df)}"
        )

    # Keep all observations from the same match together.
    match_dates = (
        df[["match_id", "date"]]
        .drop_duplicates()
        .sort_values(["date", "match_id"])
    )

    # Use approximately the first 80% of matches for training.
    split_index = int(len(match_dates) * 0.8)
    train_ids = set(match_dates.iloc[:split_index]["match_id"])
    test_ids = set(match_dates.iloc[split_index:]["match_id"])

    train = df[df["match_id"].isin(train_ids)].copy()
    test = df[df["match_id"].isin(test_ids)].copy()

    # Exclude identifiers, dates, team names and the response.
    excluded = {
        "match_id", "date", "team_1", "team_2",
        "team", "opponent", "goal_difference", "goals_scored"
    }

    predictors = [col for col in df.columns if col not in excluded]

    if len(predictors) != 8:
        raise ValueError(
            f"{task_name}: expected 8 predictors, found {len(predictors)}"
        )

    X_train = train[predictors]
    X_test = test[predictors]
    y_train = train[target]
    y_test = test[target]

    models = {
        "Baseline (Mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": make_pipeline(
            StandardScaler(), LinearRegression()
        ),
        "Ridge": make_pipeline(
            StandardScaler(), Ridge(alpha=1.0)
        ),
        "Lasso": make_pipeline(
            StandardScaler(), Lasso(alpha=0.1, max_iter=20000)
        ),
    }

    rows = []
    predictions = test[["match_id", "date"]].copy()
    predictions["actual"] = y_test.to_numpy()

    print(f"\n{'=' * 55}")
    print(task_name)
    print(f"{'=' * 55}")
    print(f"Training observations: {len(train)}")
    print(f"Testing observations: {len(test)}")
    print(f"Predictors ({len(predictors)}): {predictors}")

    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)

        mae = mean_absolute_error(y_test, predicted)
        rmse = np.sqrt(mean_squared_error(y_test, predicted))
        r2 = r2_score(y_test, predicted)

        rows.append({
            "task": task_name,
            "model": name,
            "train_rows": len(train),
            "test_rows": len(test),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
        })

        safe_name = (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        predictions[safe_name] = predicted

    scores = pd.DataFrame(rows).sort_values("MAE")
    print("\nModel comparison:")
    print(scores.to_string(index=False))

    task_code = task_name.replace(" ", "_").replace(".", "_")
    scores.to_csv(RESULTS / f"{task_code}_scores.csv", index=False)
    predictions.to_csv(
        RESULTS / f"{task_code}_predictions.csv", index=False
    )

    return scores


scores_21 = evaluate_task(
    filename="regression_2_1.csv",
    target="goal_difference",
    task_name="Regression 2.1",
    expected_rows=104,
)

scores_22 = evaluate_task(
    filename="regression_2_2.csv",
    target="goals_scored",
    task_name="Regression 2.2",
    expected_rows=208,
)

print("\nBoth regression tasks completed.")
print(f"Results saved in: {RESULTS}")
