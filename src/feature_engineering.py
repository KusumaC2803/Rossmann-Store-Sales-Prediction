import pandas as pd


def create_features(df):
    df = df.copy()

    # Make sure data is ordered correctly
    df = df.sort_values(
        ["Store", "Date"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    # Weekend indicator
    df["IsWeekend"] = (
        df["DayOfWeek"] >= 5
    ).astype(int)

    # --------------------------------------------------
    # PROMOTION FEATURES
    # --------------------------------------------------

    df["Promo_Competition"] = (
        df["Promo"]
        * (df["CompetitionDistance"] > 0).astype(int)
    )

    df["Promo2_Active"] = (
        df["Promo2"]
        * (df["Month"].isin([3, 6, 9, 12])).astype(int)
    )

    # --------------------------------------------------
    # STORE AGE / COMPETITION FEATURES
    # --------------------------------------------------

    df["CompetitionOpenMonths"] = 0

    valid_competition = (
        (df["CompetitionOpenSinceYear"] > 0)
        & (df["CompetitionOpenSinceMonth"] > 0)
    )

    df.loc[valid_competition, "CompetitionOpenMonths"] = (
        (
            df.loc[valid_competition, "Year"]
            - df.loc[valid_competition, "CompetitionOpenSinceYear"]
        ) * 12
        +
        (
            df.loc[valid_competition, "Month"]
            - df.loc[valid_competition, "CompetitionOpenSinceMonth"]
        )
    )

    df["CompetitionOpenMonths"] = (
        df["CompetitionOpenMonths"]
        .clip(lower=0)
    )

    # --------------------------------------------------
    # LOG TRANSFORMATION FOR SALES
    # --------------------------------------------------

    # Used only for historical lag/rolling calculations.
    # We will not use future sales.
    df["SalesLog"] = (
        df["Sales"] + 1
    ).apply(lambda x: __import__("math").log(x))

    # --------------------------------------------------
    # LAG FEATURES
    # --------------------------------------------------

    # IMPORTANT:
    # shift() ensures we only use past sales.
    #
    # 1 day ago
    df["Sales_Lag_1"] = (
        df.groupby("Store")["Sales"]
        .shift(1)
    )

    # 7 days ago
    df["Sales_Lag_7"] = (
        df.groupby("Store")["Sales"]
        .shift(7)
    )

    # 14 days ago
    df["Sales_Lag_14"] = (
        df.groupby("Store")["Sales"]
        .shift(14)
    )

    # 28 days ago
    df["Sales_Lag_28"] = (
        df.groupby("Store")["Sales"]
        .shift(28)
    )

    # --------------------------------------------------
    # ROLLING FEATURES
    # --------------------------------------------------

    # Shift by one day first so the current day's
    # sales are NEVER included.
    previous_sales = (
        df.groupby("Store")["Sales"]
        .shift(1)
    )

    df["Sales_Rolling_7"] = (
        previous_sales
        .groupby(df["Store"])
        .transform(
            lambda x: x.rolling(
                window=7,
                min_periods=1
            ).mean()
        )
    )

    df["Sales_Rolling_14"] = (
        previous_sales
        .groupby(df["Store"])
        .transform(
            lambda x: x.rolling(
                window=14,
                min_periods=1
            ).mean()
        )
    )

    df["Sales_Rolling_28"] = (
        previous_sales
        .groupby(df["Store"])
        .transform(
            lambda x: x.rolling(
                window=28,
                min_periods=1
            ).mean()
        )
    )

    # --------------------------------------------------
    # REMOVE ORIGINAL DATE
    # --------------------------------------------------

    df = df.drop(
        columns=["Date", "SalesLog"],
        errors="ignore"
    )

    return df