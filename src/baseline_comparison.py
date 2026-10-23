
import os

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import create_features
from src.model import build_model


LAG_DAYS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]


def add_forecast_features(row, history):
    """Create lag and rolling features using historical sales only."""

    row = row.copy()

    store_id = row["Store"]
    prediction_date = pd.Timestamp(row["Date"])
    store_history = history.get(store_id, {})

    # Lag features use the exact previous calendar dates.
    for lag in LAG_DAYS:
        previous_date = prediction_date - pd.Timedelta(days=lag)

        row[f"Sales_Lag_{lag}"] = store_history.get(
            previous_date, np.nan
        )

    # Rolling averages use available sales from previous dates.
    for window in ROLLING_WINDOWS:
        previous_sales = []

        for offset in range(1, window + 1):
            previous_date = (
                prediction_date - pd.Timedelta(days=offset)
            )

            value = store_history.get(previous_date)

            if value is not None:
                previous_sales.append(value)

        row[f"Sales_Rolling_{window}"] = (
            np.mean(previous_sales)
            if previous_sales
            else np.nan
        )

    return row


def build_history(training_data):
    """Create a store/date lookup of historical sales."""

    history = {}

    for store_id, date, sales in training_data[
        ["Store", "Date", "Sales"]
    ].itertuples(index=False, name=None):

        history.setdefault(store_id, {})[
            pd.Timestamp(date)
        ] = float(sales)

    return history


def recursive_forecast(
    model,
    training_data,
    validation_data,
    feature_columns,
):
    """Forecast validation dates sequentially without actual-sales leakage."""

    history = build_history(training_data)
    predictions = []

    validation_dates = sorted(
        validation_data["Date"].unique()
    )

    for position, date in enumerate(validation_dates, start=1):

        daily_data = validation_data[
            validation_data["Date"] == date
        ].copy()

        forecast_rows = [
            add_forecast_features(row, history)
            for _, row in daily_data.iterrows()
        ]

        forecast_data = pd.DataFrame(forecast_rows)

        X_daily = forecast_data.reindex(
            columns=feature_columns
        )

        daily_predictions = np.maximum(
            model.predict(X_daily),
            0,
        )

        # Save predictions into history for future forecast dates.
        for store_id, predicted_sales in zip(
            forecast_data["Store"],
            daily_predictions,
        ):
            history.setdefault(store_id, {})[
                pd.Timestamp(date)
            ] = float(predicted_sales)

        predictions.append(
            pd.DataFrame(
                {
                    "Store": forecast_data["Store"].to_numpy(),
                    "Date": forecast_data["Date"].to_numpy(),
                    "ActualSales": daily_data["Sales"].to_numpy(),
                    "PredictedSales": daily_predictions,
                }
            )
        )

        if position % 30 == 0 or position == len(validation_dates):
            print(
                f"Forecasted {position}/"
                f"{len(validation_dates)} validation dates"
            )

    if not predictions:
        raise ValueError("No validation predictions were generated.")

    return pd.concat(predictions, ignore_index=True)


def run_recursive_validation(save_outputs=True):
    """
    Train on the earlier period and evaluate recursive forecasts
    on the later period.

    Returns:
        dict: Validation MAE and RMSE.
    """

    print("=" * 65)
    print("ROSSMANN SALES: RECURSIVE FUTURE-FORECAST VALIDATION")
    print("=" * 65)

    # 1. Load and preprocess.
    print("\n[1/6] Loading datasets...")

    train, store = load_data()
    df = preprocess_data(train, store)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        ["Store", "Date"]
    ).reset_index(drop=True)

    # Preserve zero-sales rows for historical features.

    # 2. Split by date.
    print("\n[2/6] Splitting chronologically...")

    all_dates = np.sort(df["Date"].dropna().unique())

    if len(all_dates) < 2:
        raise ValueError("Not enough unique dates to create a split.")

    split_position = int(len(all_dates) * 0.80)
    split_position = max(
        1,
        min(split_position, len(all_dates) - 1),
    )

    validation_start = pd.Timestamp(all_dates[split_position])

    training = df[df["Date"] < validation_start].copy()
    validation = df[df["Date"] >= validation_start].copy()

    print("Training rows:", len(training))
    print("Validation rows:", len(validation))
    print("Training ends:", training["Date"].max().date())
    print("Validation starts:", validation["Date"].min().date())
    print("Validation ends:", validation["Date"].max().date())

    # 3. Create training features.
    print("\n[3/6] Creating training features...")

    training_features = create_features(training)

    lag_columns = [
        "Sales_Lag_1",
        "Sales_Lag_7",
        "Sales_Lag_14",
        "Sales_Lag_28",
    ]

    training_features = training_features.dropna(
        subset=lag_columns
    ).copy()

    if training_features.empty:
        raise ValueError(
            "No training rows remain after lag filtering."
        )

    X_train = training_features.drop(
        columns=[
            "Sales",
            "Customers",
            "SalesLog",
            "Date",
        ],
        errors="ignore",
    )

    y_train = training_features["Sales"]

    # 4. Train.
    print("\n[4/6] Training Random Forest...")

    model = build_model(X_train)
    model.fit(X_train, y_train)

    feature_columns = X_train.columns.tolist()

    # 5. Forecast recursively.
    print("\n[5/6] Forecasting validation period...")

    # Static features are created from the validation data.
    # Lag and rolling features are overwritten before prediction.
    validation_features = create_features(validation)

    forecast_results = recursive_forecast(
        model=model,
        training_data=training,
        validation_data=validation_features,
        feature_columns=feature_columns,
    )

    # 6. Evaluate.
    print("\n[6/6] Evaluating forecast...")

    actual = forecast_results["ActualSales"].to_numpy()
    predicted = forecast_results["PredictedSales"].to_numpy()

    rf_mae = mean_absolute_error(actual, predicted)
    rf_rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    baseline_predictions = np.full(
        len(actual),
        training["Sales"].mean(),
    )

    baseline_mae = mean_absolute_error(
        actual,
        baseline_predictions,
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(actual, baseline_predictions)
    )

    results = pd.DataFrame(
        [
            {
                "Model": "Mean Baseline",
                "MAE": baseline_mae,
                "RMSE": baseline_rmse,
            },
            {
                "Model": "Random Forest Recursive Forecast",
                "MAE": rf_mae,
                "RMSE": rf_rmse,
            },
        ]
    )

    print("\nModel Comparison:")
    print(
        results.to_string(
            index=False,
            float_format="%.2f",
        )
    )

    if save_outputs:
        os.makedirs("outputs", exist_ok=True)

        results.to_csv(
            "outputs/baseline_comparison.csv",
            index=False,
        )

        forecast_results.to_csv(
            "outputs/recursive_validation_predictions.csv",
            index=False,
        )

        print("\nSaved:")
        print("- outputs/baseline_comparison.csv")
        print("- outputs/recursive_validation_predictions.csv")

    return {
        "MAE": float(rf_mae),
        "RMSE": float(rf_rmse),
        "Baseline_MAE": float(baseline_mae),
        "Baseline_RMSE": float(baseline_rmse),
    }


def main():
    run_recursive_validation(save_outputs=True)


if __name__ == "__main__":
    main()

