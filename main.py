import os
import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import create_features
from src.model import build_model


def calculate_rmspe(y_true, y_pred):
    """
    Calculate Root Mean Square Percentage Error.

    Zero actual sales values are excluded to avoid
    division by zero.
    """

    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    non_zero = y_true != 0

    percentage_errors = (
        (y_true[non_zero] - y_pred[non_zero])
        / y_true[non_zero]
    )

    rmspe = (
        (percentage_errors ** 2).mean()
    ) ** 0.5

    return rmspe


def main():

    print("=" * 60)
    print("Rossmann Store Sales Prediction")
    print("=" * 60)

    # ==================================================
    # 1. LOAD DATA
    # ==================================================

    print("\n[1/6] Loading datasets...")

    train, store = load_data()

    print("Train shape:", train.shape)
    print("Store shape:", store.shape)

    # ==================================================
    # 2. PREPROCESS DATA
    # ==================================================

    print("\n[2/6] Preprocessing data...")

    df = preprocess_data(
        train,
        store
    )

    # Remove days with zero sales.
    # These are generally closed-store observations.
    df = df[
        df["Sales"] > 0
    ].copy()

    print("Processed shape:", df.shape)

    # ==================================================
    # 3. FEATURE ENGINEERING
    # ==================================================

    print("\n[3/6] Creating time-series features...")

    df = create_features(df)

    # --------------------------------------------------
    # Remove rows where historical features are missing.
    #
    # The first 28 observations for a store cannot have
    # a complete 28-day lag history.
    # --------------------------------------------------

    lag_columns = [
        "Sales_Lag_1",
        "Sales_Lag_7",
        "Sales_Lag_14",
        "Sales_Lag_28",
    ]

    df = df.dropna(
        subset=lag_columns
    ).copy()

    print(
        "Dataset after time-series features:",
        df.shape
    )

    # ==================================================
    # 4. PREPARE TRAINING DATA
    # ==================================================

    print("\n[4/6] Preparing training data...")

    # Sales = target
    #
    # Customers is deliberately removed because future
    # test.csv does not contain actual customer counts.
    X = df.drop(
        columns=[
            "Sales",
            "Customers"
        ]
    )

    y = df["Sales"]

    # --------------------------------------------------
    # Chronological split
    #
    # First 80% = training
    # Last 20% = validation
    #
    # We DO NOT randomly shuffle time-series data.
    # --------------------------------------------------

    split_index = int(
        len(X) * 0.80
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Validation samples:",
        len(X_test)
    )

    # ==================================================
    # 5. TRAIN MODEL
    # ==================================================

    print(
        "\n[5/6] Training Random Forest model..."
    )

    model = build_model(
        X_train
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Model training completed."
    )

    # ==================================================
    # 6. EVALUATE MODEL
    # ==================================================

    print(
        "\n[6/6] Evaluating model..."
    )

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------
    # MAE
    # --------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    # --------------------------------------------------
    # MSE
    # --------------------------------------------------

    mse = mean_squared_error(
        y_test,
        predictions
    )

    # --------------------------------------------------
    # RMSE
    # --------------------------------------------------

    rmse = mse ** 0.5

    # --------------------------------------------------
    # R²
    # --------------------------------------------------

    r2 = r2_score(
        y_test,
        predictions
    )

    # --------------------------------------------------
    # RMSPE
    # --------------------------------------------------

    rmspe = calculate_rmspe(
        y_test,
        predictions
    )

    # ==================================================
    # DISPLAY RESULTS
    # ==================================================

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(
        f"MAE   : {mae:,.2f}"
    )

    print(
        f"RMSE  : {rmse:,.2f}"
    )

    print(
        f"R²    : {r2:.4f}"
    )

    print(
        f"RMSPE : {rmspe:.4f}"
    )

    # ==================================================
    # SAVE MODEL
    # ==================================================

    print("\nSaving model...")

    os.makedirs(
        "models",
        exist_ok=True
    )

    model_path = (
        "models/random_forest_timeseries.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "Model saved to:",
        model_path
    )

    # ==================================================
    # SAVE PREDICTIONS
    # ==================================================

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    predictions_df = pd.DataFrame({
        "Actual_Sales": y_test.values,
        "Predicted_Sales": predictions
    })

    predictions_path = (
        "outputs/timeseries_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    print(
        "Predictions saved to:",
        predictions_path
    )

    # ==================================================
    # SAVE METRICS
    # ==================================================

    metrics_df = pd.DataFrame({
        "Metric": [
            "MAE",
            "RMSE",
            "R2",
            "RMSPE"
        ],
        "Value": [
            mae,
            rmse,
            r2,
            rmspe
        ]
    })

    metrics_path = (
        "outputs/timeseries_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    print(
        "Metrics saved to:",
        metrics_path
    )

    # ==================================================
    # COMPLETION
    # ==================================================

    print("\n" + "=" * 60)
    print(
        "TIME-SERIES PIPELINE COMPLETED SUCCESSFULLY"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()