
import os
import joblib
import pandas as pd

from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import create_features
from src.model import build_model


def main():

    print("=" * 65)
    print("ROSSMANN STORE SALES PREDICTION")
    print("FINAL MODEL TRAINING")
    print("=" * 65)

    # ==================================================
    # 1. LOAD DATA
    # ==================================================

    print("\n[1/5] Loading datasets...")

    train, store = load_data()

    print("Train shape:", train.shape)
    print("Store shape:", store.shape)

    # ==================================================
    # 2. PREPROCESS DATA
    # ==================================================

    print("\n[2/5] Preprocessing data...")

    df = preprocess_data(train, store)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        ["Store", "Date"]
    ).reset_index(drop=True)

    # Keep zero-sales observations because they are
    # important for constructing historical features.

    print("Processed shape:", df.shape)

    # ==================================================
    # 3. CREATE FEATURES
    # ==================================================

    print("\n[3/5] Creating time-series features...")

    df = create_features(df)

    lag_columns = [
        "Sales_Lag_1",
        "Sales_Lag_7",
        "Sales_Lag_14",
        "Sales_Lag_28",
    ]

    # Training rows need all four lag features.
    # This does not remove the original historical rows
    # before feature engineering.

    df = df.dropna(
        subset=lag_columns
    ).copy()

    print("Training rows after lag filtering:", len(df))

    if df.empty:
        raise ValueError(
            "No training rows remain after feature engineering."
        )

    # ==================================================
    # 4. PREPARE AND TRAIN MODEL
    # ==================================================

    print("\n[4/5] Training final Random Forest model...")

    X_train = df.drop(
        columns=[
            "Sales",
            "Customers",
            "SalesLog",
            "Date",
        ],
        errors="ignore",
    )

    y_train = df["Sales"]

    model = build_model(X_train)

    model.fit(
        X_train,
        y_train,
    )

    print("Final model training completed.")

    # ==================================================
    # 5. SAVE MODEL AND TRAINING SUMMARY
    # ==================================================

    print("\n[5/5] Saving model and training summary...")

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    model_path = "models/random_forest_timeseries.pkl"

    joblib.dump(
        model,
        model_path,
    )

    print("Model saved to:", model_path)

    training_summary = pd.DataFrame(
        {
            "Item": [
                "Training rows",
                "Training start date",
                "Training end date",
                "Number of features",
                "Model",
            ],
            "Value": [
                len(X_train),
                df["Date"].min().date(),
                df["Date"].max().date(),
                len(X_train.columns),
                "RandomForestRegressor",
            ],
        }
    )

    training_summary.to_csv(
        "outputs/training_summary.csv",
        index=False,
    )

    print(
        "Training summary saved to "
        "outputs/training_summary.csv"
    )

    print("\n" + "=" * 65)
    print("FINAL MODEL TRAINING COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()

