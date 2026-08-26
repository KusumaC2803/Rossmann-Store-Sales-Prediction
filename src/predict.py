import os
import joblib
import pandas as pd


def prepare_test_data(train, test, store):
    """
    Build prediction features for the official Rossmann test set.

    Historical sales are used to construct lag features.
    Future test rows never use their own unknown Sales values.
    """

    # --------------------------------------------------
    # Merge store information
    # --------------------------------------------------

    train_df = train.merge(
        store,
        on="Store",
        how="left"
    )

    test_df = test.merge(
        store,
        on="Store",
        how="left"
    )

    train_df["Date"] = pd.to_datetime(train_df["Date"])
    test_df["Date"] = pd.to_datetime(test_df["Date"])

    # --------------------------------------------------
    # Keep historical and future rows together
    # --------------------------------------------------

    train_df["_is_test"] = 0
    test_df["_is_test"] = 1

    # Test has no Sales
    test_df["Sales"] = pd.NA

    combined = pd.concat(
        [train_df, test_df],
        ignore_index=True
    )

    combined = combined.sort_values(
        ["Store", "Date"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    combined["CompetitionDistance"] = (
        combined["CompetitionDistance"]
        .fillna(
            combined["CompetitionDistance"].median()
        )
    )

    numeric_columns = [
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear"
    ]

    for column in numeric_columns:
        combined[column] = (
            combined[column]
            .fillna(0)
        )

    categorical_columns = [
        "StateHoliday",
        "StoreType",
        "Assortment",
        "PromoInterval"
    ]

    for column in categorical_columns:
        combined[column] = (
            combined[column]
            .fillna("None")
            .astype(str)
        )

    # --------------------------------------------------
    # Calendar features
    # --------------------------------------------------

    combined["Year"] = combined["Date"].dt.year
    combined["Month"] = combined["Date"].dt.month
    combined["Day"] = combined["Date"].dt.day

    combined["WeekOfYear"] = (
        combined["Date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    combined["Quarter"] = (
        combined["Date"].dt.quarter
    )

    combined["DayOfWeek"] = (
        combined["Date"].dt.dayofweek
    )

    combined["IsWeekend"] = (
        combined["DayOfWeek"] >= 5
    ).astype(int)

    # --------------------------------------------------
    # Promotion features
    # --------------------------------------------------

    combined["Promo_Competition"] = (
        combined["Promo"]
        * (
            combined["CompetitionDistance"] > 0
        ).astype(int)
    )

    combined["Promo2_Active"] = (
        combined["Promo2"]
        * (
            combined["Month"].isin(
                [3, 6, 9, 12]
            )
        ).astype(int)
    )

    # --------------------------------------------------
    # Competition age
    # --------------------------------------------------

    combined["CompetitionOpenMonths"] = 0

    valid = (
        (combined["CompetitionOpenSinceYear"] > 0)
        &
        (combined["CompetitionOpenSinceMonth"] > 0)
    )

    combined.loc[
        valid,
        "CompetitionOpenMonths"
    ] = (
        (
            combined.loc[
                valid, "Year"
            ]
            -
            combined.loc[
                valid,
                "CompetitionOpenSinceYear"
            ]
        ) * 12
        +
        (
            combined.loc[
                valid, "Month"
            ]
            -
            combined.loc[
                valid,
                "CompetitionOpenSinceMonth"
            ]
        )
    )

    combined["CompetitionOpenMonths"] = (
        combined["CompetitionOpenMonths"]
        .clip(lower=0)
    )

    # --------------------------------------------------
    # IMPORTANT:
    # Create lag features from actual historical
    # sales using calendar dates.
    # --------------------------------------------------

    sales_history = (
        train_df[
            ["Store", "Date", "Sales"]
        ]
        .copy()
    )

    sales_history["Date"] = pd.to_datetime(
        sales_history["Date"]
    )

    sales_history = sales_history.drop_duplicates(
        ["Store", "Date"]
    )

    # --------------------------------------------------
    # Exact date-based lag function
    # --------------------------------------------------

    def add_lag_feature(
        target_df,
        days,
        feature_name
    ):
        lookup = sales_history.copy()

        lookup["Date"] = (
            lookup["Date"]
            + pd.Timedelta(days=days)
        )

        lookup = lookup.rename(
            columns={
                "Sales": feature_name
            }
        )

        return target_df.merge(
            lookup[
                [
                    "Store",
                    "Date",
                    feature_name
                ]
            ],
            on=["Store", "Date"],
            how="left"
        )

    # --------------------------------------------------
    # Create historical lag features
    # --------------------------------------------------

    combined = add_lag_feature(
        combined,
        -1,
        "Sales_Lag_1"
    )

    combined = add_lag_feature(
        combined,
        -7,
        "Sales_Lag_7"
    )

    combined = add_lag_feature(
        combined,
        -14,
        "Sales_Lag_14"
    )

    combined = add_lag_feature(
        combined,
        -28,
        "Sales_Lag_28"
    )

    # --------------------------------------------------
    # Historical rolling averages
    #
    # Calculate from actual training observations.
    # --------------------------------------------------

    history = train_df[
        ["Store", "Date", "Sales"]
    ].copy()

    history["Date"] = pd.to_datetime(
        history["Date"]
    )

    history = history.sort_values(
        ["Store", "Date"]
    )

    history["Rolling_7"] = (
        history.groupby("Store")["Sales"]
        .transform(
            lambda x: x.shift(1).rolling(
                7,
                min_periods=1
            ).mean()
        )
    )

    history["Rolling_14"] = (
        history.groupby("Store")["Sales"]
        .transform(
            lambda x: x.shift(1).rolling(
                14,
                min_periods=1
            ).mean()
        )
    )

    history["Rolling_28"] = (
        history.groupby("Store")["Sales"]
        .transform(
            lambda x: x.shift(1).rolling(
                28,
                min_periods=1
            ).mean()
        )
    )

    # Map latest available historical rolling values
    # to test dates using the immediately preceding
    # historical observation.

    rolling_values = history[
        [
            "Store",
            "Date",
            "Rolling_7",
            "Rolling_14",
            "Rolling_28"
        ]
    ].copy()

    rolling_values["Date"] = (
        rolling_values["Date"]
        + pd.Timedelta(days=1)
    )

    combined = combined.merge(
        rolling_values,
        on=["Store", "Date"],
        how="left"
    )

    combined = combined.rename(
        columns={
            "Rolling_7": "Sales_Rolling_7",
            "Rolling_14": "Sales_Rolling_14",
            "Rolling_28": "Sales_Rolling_28"
        }
    )

    # --------------------------------------------------
    # Select test rows
    # --------------------------------------------------

    test_features = combined[
        combined["_is_test"] == 1
    ].copy()

    # --------------------------------------------------
    # Remove prediction-only columns
    # --------------------------------------------------

    test_features = test_features.drop(
        columns=[
            "Id",
            "Sales",
            "Customers",
            "Date",
            "_is_test"
        ],
        errors="ignore"
    )

    return test_features


def generate_predictions():

    print("=" * 60)
    print("ROSSMANN FUTURE SALES PREDICTION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------

    print("\n[1/4] Loading data...")

    train = pd.read_csv(
        "data/train.csv",
        low_memory=False
    )

    test = pd.read_csv(
        "data/test.csv",
        low_memory=False
    )

    store = pd.read_csv(
        "data/store.csv",
        low_memory=False
    )

    print("Train:", train.shape)
    print("Test :", test.shape)
    print("Store:", store.shape)

    # --------------------------------------------------
    # 2. Load trained model
    # --------------------------------------------------

    print("\n[2/4] Loading trained model...")

    model_path = (
        "models/random_forest_timeseries.pkl"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = joblib.load(
        model_path
    )

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------
    # 3. Prepare future features
    # --------------------------------------------------

    print(
        "\n[3/4] Preparing future test features..."
    )

    X_test = prepare_test_data(
        train,
        test,
        store
    )

    print(
        "Test feature shape:",
        X_test.shape
    )

    # --------------------------------------------------
    # Check lag availability
    # --------------------------------------------------

    lag_columns = [
        "Sales_Lag_1",
        "Sales_Lag_7",
        "Sales_Lag_14",
        "Sales_Lag_28"
    ]

    print("\nMissing lag values:")

    print(
        X_test[lag_columns]
        .isna()
        .sum()
    )

    # --------------------------------------------------
    # Fill any remaining unavailable values
    # --------------------------------------------------

    X_test = X_test.fillna(0)

    # --------------------------------------------------
    # 4. Predict
    # --------------------------------------------------

    print(
        "\n[4/4] Generating predictions..."
    )

    predictions = model.predict(
        X_test
    )

    predictions = predictions.clip(
        min=0
    )

    # --------------------------------------------------
    # Create submission
    # --------------------------------------------------

    submission = pd.DataFrame({
        "Id": test["Id"],
        "Sales": predictions
    })

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    submission_path = (
        "outputs/submission.csv"
    )

    submission.to_csv(
        submission_path,
        index=False
    )

    print(
        "\nSubmission saved to:",
        submission_path
    )

    print("\nPrediction summary:")

    print(
        submission["Sales"].describe()
    )

    print("\nFirst 10 predictions:")

    print(
        submission.head(10)
    )

    print("\n" + "=" * 60)
    print(
        "FUTURE FORECASTING COMPLETED"
    )
    print("=" * 60)


if __name__ == "__main__":
    generate_predictions()