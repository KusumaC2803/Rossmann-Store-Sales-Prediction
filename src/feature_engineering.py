
import numpy as np
import pandas as pd


def create_features(df):
    df = df.copy()

    # Keep Date for chronological validation and forecasting.
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort by store and date before calculating historical features.
    df = df.sort_values(
        ["Store", "Date"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 1. DATE FEATURES
    # --------------------------------------------------

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = (
        df["Date"].dt.isocalendar().week.astype(int)
    )
    df["Quarter"] = df["Date"].dt.quarter
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    df["IsWeekend"] = (
        df["DayOfWeek"] >= 5
    ).astype(int)

    # --------------------------------------------------
    # 2. PROMOTION FEATURES
    # --------------------------------------------------

    df["Promo_Competition"] = (
        df["Promo"]
        * (df["CompetitionDistance"] > 0).astype(int)
    )

    # Use each store's actual PromoInterval.
    # Examples: "Jan,Apr,Jul,Oct" and "Feb,May,Aug,Nov".
    month_names = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }

    current_month = df["Month"].map(month_names)

    if "PromoInterval" in df.columns:
        interval = (
            df["PromoInterval"]
            .fillna("")
            .astype(str)
        )

        month_is_active = pd.Series(
            [
                month in [
                    item.strip()
                    for item in months.split(",")
                    if item.strip()
                ]
                for month, months in zip(
                    current_month, interval
                )
            ],
            index=df.index,
        )

        # Promo2SinceWeek is a week number, not a month.
        # Calculate the start date from the ISO year and week.
        promo2_start = pd.to_datetime(
            df["Promo2SinceYear"].astype("Int64").astype(str)
            + "-W"
            + df["Promo2SinceWeek"].astype("Int64").astype(str)
            + "-1",
            format="%G-W%V-%u",
            errors="coerce",
        )

        has_promo2_start = (
            df["Promo2SinceYear"].fillna(0).gt(0)
            & df["Promo2SinceWeek"].fillna(0).gt(0)
        )

        promo2_started = (
            has_promo2_start
            & promo2_start.notna()
            & (df["Date"] >= promo2_start)
        )

        df["Promo2_Active"] = (
            df["Promo2"].fillna(0).eq(1)
            & month_is_active
            & promo2_started
        ).astype(int)

    else:
        df["Promo2_Active"] = 0

    # --------------------------------------------------
    # 3. COMPETITION FEATURES
    # --------------------------------------------------

    competition_year = (
        df["CompetitionOpenSinceYear"].fillna(0)
    )
    competition_month = (
        df["CompetitionOpenSinceMonth"].fillna(0)
    )

    df["CompetitionOpenMonths"] = np.where(
        (competition_year > 0)
        & (competition_month > 0),
        (
            (df["Year"] - competition_year) * 12
            + (df["Month"] - competition_month)
        ),
        0,
    )

    df["CompetitionOpenMonths"] = (
        df["CompetitionOpenMonths"]
        .clip(lower=0)
    )

    # --------------------------------------------------
    # 4. SALES LAG FEATURES
    # --------------------------------------------------

    # These features use previous rows for each store.
    # They are appropriate when earlier observed sales
    # are available at prediction time.
    store_sales = df.groupby("Store")["Sales"]

    df["Sales_Lag_1"] = store_sales.shift(1)
    df["Sales_Lag_7"] = store_sales.shift(7)
    df["Sales_Lag_14"] = store_sales.shift(14)
    df["Sales_Lag_28"] = store_sales.shift(28)

    # --------------------------------------------------
    # 5. ROLLING SALES FEATURES
    # --------------------------------------------------

    previous_sales = df.groupby("Store")["Sales"].shift(1)

    df["Sales_Rolling_7"] = (
        previous_sales.groupby(df["Store"])
        .transform(
            lambda x: x.rolling(
                window=7,
                min_periods=1,
            ).mean()
        )
    )

    df["Sales_Rolling_14"] = (
        previous_sales.groupby(df["Store"])
        .transform(
            lambda x: x.rolling(
                window=14,
                min_periods=1,
            ).mean()
        )
    )

    df["Sales_Rolling_28"] = (
        previous_sales.groupby(df["Store"])
        .transform(
            lambda x: x.rolling(
                window=28,
                min_periods=1,
            ).mean()
        )
    )

    # Do not remove Date here.
    # The validation script needs it for chronological splitting.

    return df

