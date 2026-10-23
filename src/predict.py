import os
import joblib
import numpy as np
import pandas as pd


def add_static_features(df):
    """Create calendar, promotion, and competition features."""
    df = df.copy()

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    df["Promo"] = (
        pd.to_numeric(df["Promo"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    if "CompetitionDistance" in df.columns:
        distance = pd.to_numeric(
            df["CompetitionDistance"], errors="coerce"
        )
        df["Promo_Competition"] = (
            (df["Promo"] == 1) & (distance > 0)
        ).astype(int)
    else:
        df["Promo_Competition"] = 0

    # Match the Promo2 logic in src/feature_engineering.py.
    df["Promo2_Active"] = 0

    required_promo2 = [
        "Promo2",
        "PromoInterval",
        "Promo2SinceYear",
        "Promo2SinceWeek",
    ]

    if all(col in df.columns for col in required_promo2):
        promo2 = pd.to_numeric(
            df["Promo2"], errors="coerce"
        ).fillna(0)

        intervals = (
            df["PromoInterval"].fillna("").astype(str).str.upper()
        )
        month_names = df["Date"].dt.strftime("%b").str.upper()

        interval_active = pd.Series(
            [
                month in [
                    item.strip()
                    for item in interval.split(",")
                    if item.strip()
                ]
                for month, interval in zip(month_names, intervals)
            ],
            index=df.index,
        )

        start_year = pd.to_numeric(
            df["Promo2SinceYear"], errors="coerce"
        )
        start_week = pd.to_numeric(
            df["Promo2SinceWeek"], errors="coerce"
        )

        valid_start = (
            start_year.notna()
            & start_week.notna()
            & start_year.gt(0)
            & start_week.between(1, 53)
        )

        start_date = pd.Series(
            pd.NaT, index=df.index, dtype="datetime64[ns]"
        )

        if valid_start.any():
            iso_dates = (
                start_year.loc[valid_start].astype(int).astype(str)
                + "-W"
                + start_week.loc[valid_start]
                .astype(int)
                .astype(str)
                .str.zfill(2)
                + "-1"
            )
            start_date.loc[valid_start] = pd.to_datetime(
                iso_dates,
                format="%G-W%V-%u",
                errors="coerce",
            )

        df["Promo2_Active"] = (
            promo2.eq(1)
            & interval_active
            & start_date.notna()
            & (df["Date"] >= start_date)
        ).astype(int)

    df["CompetitionOpenMonths"] = 0

    comp_columns = [
        "CompetitionOpenSinceYear",
        "CompetitionOpenSinceMonth",
    ]

    if all(col in df.columns for col in comp_columns):
        comp_year = pd.to_numeric(
            df["CompetitionOpenSinceYear"], errors="coerce"
        )
        comp_month = pd.to_numeric(
            df["CompetitionOpenSinceMonth"], errors="coerce"
        )

        valid = (
            comp_year.notna()
            & comp_month.notna()
            & comp_year.gt(0)
            & comp_month.between(1, 12)
        )

        months_open = (
            (df["Year"] - comp_year) * 12
            + (df["Month"] - comp_month)
        )

        df.loc[valid, "CompetitionOpenMonths"] = (
            months_open.loc[valid].clip(lower=0).astype(int)
        )

    return df


def prepare_test_data(train, test, store, model):
    """
    Generate future test features chronologically.

    Actual training sales are used as history. Predictions from earlier
    test dates are added to history for later dates, so unknown future
    sales are never used as inputs.
    """
    train_df = train.merge(store, on="Store", how="left")
    test_df = test.merge(store, on="Store", how="left")

    train_df["Date"] = pd.to_datetime(train_df["Date"], errors="raise")
    test_df["Date"] = pd.to_datetime(test_df["Date"], errors="raise")

    test_df["_original_order"] = np.arange(len(test_df))

    # Fill store metadata consistently across training and test rows.
    combined_metadata = pd.concat(
        [
            train_df[["Store", "CompetitionDistance"]],
            test_df[["Store", "CompetitionDistance"]],
        ],
        ignore_index=True,
    )
    distance_median = combined_metadata["CompetitionDistance"].median()

    for frame in (train_df, test_df):
        if "CompetitionDistance" in frame.columns:
            frame["CompetitionDistance"] = (
                frame["CompetitionDistance"].fillna(distance_median)
            )

        for column in [
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear",
            "Promo2SinceWeek",
            "Promo2SinceYear",
        ]:
            if column in frame.columns:
                frame[column] = frame[column].fillna(0)

        for column in [
            "StateHoliday",
            "StoreType",
            "Assortment",
            "PromoInterval",
        ]:
            if column in frame.columns:
                frame[column] = (
                    frame[column].fillna("None").astype(str)
                )

    test_df = add_static_features(test_df)
    test_df = test_df.sort_values(
        ["Date", "Store"]
    ).reset_index(drop=True)

    # History is keyed by (store, date). Start with actual training sales.
    history = {}
    historical_rows = train_df[["Store", "Date", "Sales"]].dropna(
        subset=["Sales"]
    )

    for row in historical_rows.itertuples(index=False):
        history[(int(row.Store), pd.Timestamp(row.Date))] = float(row.Sales)

    lag_days = [1, 7, 14, 28]
    rolling_windows = [7, 14, 28]
    prediction_records = []

    # Predict one date at a time. All stores for a date are predicted
    # before that date's predictions are added to history.
    for date in sorted(test_df["Date"].unique()):
        date = pd.Timestamp(date)
        day_df = test_df.loc[test_df["Date"] == date].copy()

        for lag in lag_days:
            feature_name = f"Sales_Lag_{lag}"
            previous_date = date - pd.Timedelta(days=lag)
            day_df[feature_name] = [
                history.get((int(store_id), previous_date), np.nan)
                for store_id in day_df["Store"]
            ]

        for window in rolling_windows:
            feature_name = f"Sales_Rolling_{window}"
            rolling_values = []

            for store_id in day_df["Store"]:
                previous_sales = [
                    history.get(
                        (
                            int(store_id),
                            date - pd.Timedelta(days=offset),
                        ),
                        np.nan,
                    )
                    for offset in range(1, window + 1)
                ]

                if all(pd.notna(value) for value in previous_sales):
                    rolling_values.append(float(np.mean(previous_sales)))
                else:
                    rolling_values.append(np.nan)

            day_df[feature_name] = rolling_values

        # Match the training target/features: no Id, Date, Sales,
        # Customers, or helper columns are passed to the model.
        X_day = day_df.drop(
            columns=[
                "Id",
                "Date",
                "Sales",
                "Customers",
                "_original_order",
            ],
            errors="ignore",
        )

        # The fitted pipeline handles remaining missing values.
        day_predictions = np.asarray(model.predict(X_day)).clip(min=0)

        for store_id, original_order, prediction in zip(
            day_df["Store"],
            day_df["_original_order"],
            day_predictions,
        ):
            prediction_records.append(
                {
                    "_original_order": int(original_order),
                    "Store": int(store_id),
                    "Date": date,
                    "Predicted_Sales": float(prediction),
                }
            )

        # Add this date's predictions to history for later test dates.
        for store_id, prediction in zip(
            day_df["Store"], day_predictions
        ):
            history[(int(store_id), date)] = float(prediction)

    predictions_df = pd.DataFrame(prediction_records).sort_values(
        "_original_order"
    )

    return predictions_df["Predicted_Sales"].to_numpy()


def generate_predictions():
    print("=" * 60)
    print("ROSSMANN FUTURE SALES PREDICTION")
    print("=" * 60)

    print("\n[1/4] Loading datasets...")

    train = pd.read_csv("data/train.csv", low_memory=False)
    test = pd.read_csv("data/test.csv", low_memory=False)
    store = pd.read_csv("data/store.csv", low_memory=False)

    print("Train:", train.shape)
    print("Test :", test.shape)
    print("Store:", store.shape)

    print("\n[2/4] Loading trained model...")

    model_path = "models/random_forest_timeseries.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run main.py first."
        )

    model = joblib.load(model_path)
    print("Model loaded successfully.")

    print("\n[3/4] Generating chronological predictions...")

    predictions = prepare_test_data(train, test, store, model)

    if len(predictions) != len(test):
        raise ValueError(
            "Prediction count does not match the number of test rows."
        )

    submission = pd.DataFrame(
        {
            "Id": test["Id"],
            "Sales": predictions,
        }
    )

    os.makedirs("outputs", exist_ok=True)
    submission_path = "outputs/submission.csv"
    submission.to_csv(submission_path, index=False)

    print("\n[4/4] Forecast completed.")
    print("Submission saved to:", submission_path)
    print("\nPrediction summary:")
    print(submission["Sales"].describe())
    print("\nFirst 10 predictions:")
    print(submission.head(10))

    print("\n" + "=" * 60)
    print("PREDICTION PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    generate_predictions()
