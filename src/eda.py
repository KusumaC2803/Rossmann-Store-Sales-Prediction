import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda():

    print("=" * 60)
    print("ROSSMANN SALES EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    print("\n[1/5] Loading datasets...")

    train = pd.read_csv(
        "data/train.csv",
        low_memory=False
    )

    store = pd.read_csv(
        "data/store.csv",
        low_memory=False
    )

    train["Date"] = pd.to_datetime(
        train["Date"]
    )

    # Merge store information
    df = train.merge(
        store,
        on="Store",
        how="left"
    )

    # Keep open stores for sales analysis
    df = df[df["Sales"] > 0].copy()

    os.makedirs(
        "outputs/eda",
        exist_ok=True
    )

    print(
        "Dataset shape:",
        df.shape
    )

    # --------------------------------------------------
    # Date features
    # --------------------------------------------------

    print("\n[2/5] Creating analysis features...")

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    # --------------------------------------------------
    # 1. Daily sales trend
    # --------------------------------------------------

    print("\n[3/5] Creating sales trend...")

    daily_sales = (
        df.groupby("Date")["Sales"]
        .mean()
        .reset_index()
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        daily_sales["Date"],
        daily_sales["Sales"]
    )

    plt.title(
        "Average Daily Rossmann Sales"
    )

    plt.xlabel("Date")
    plt.ylabel("Average Sales")

    plt.tight_layout()

    plt.savefig(
        "outputs/eda/daily_sales_trend.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # 2. Sales by day of week
    # --------------------------------------------------

    print(
        "[4/5] Creating weekday analysis..."
    )

    weekday_sales = (
        df.groupby("DayOfWeek")["Sales"]
        .mean()
    )

    weekday_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    weekday_sales.index = [
        weekday_names[i]
        for i in weekday_sales.index
    ]

    plt.figure(
        figsize=(10, 6)
    )

    weekday_sales.plot(
        kind="bar"
    )

    plt.title(
        "Average Sales by Day of Week"
    )

    plt.xlabel("Day of Week")
    plt.ylabel("Average Sales")

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/eda/sales_by_weekday.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # 3. Promotion impact
    # --------------------------------------------------

    promo_sales = (
        df.groupby("Promo")["Sales"]
        .mean()
        .reset_index()
    )

    promo_sales["Promo"] = (
        promo_sales["Promo"]
        .map({
            0: "No Promotion",
            1: "Promotion"
        })
    )

    plt.figure(
        figsize=(8, 6)
    )

    sns.barplot(
        data=promo_sales,
        x="Promo",
        y="Sales"
    )

    plt.title(
        "Average Sales: Promotion vs No Promotion"
    )

    plt.xlabel("Promotion Status")
    plt.ylabel("Average Sales")

    plt.tight_layout()

    plt.savefig(
        "outputs/eda/promotion_impact.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # 4. Store type analysis
    # --------------------------------------------------

    print(
        "[5/5] Creating store-type analysis..."
    )

    store_type_sales = (
        df.groupby("StoreType")["Sales"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    plt.figure(
        figsize=(8, 6)
    )

    store_type_sales.plot(
        kind="bar"
    )

    plt.title(
        "Average Sales by Store Type"
    )

    plt.xlabel("Store Type")
    plt.ylabel("Average Sales")

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/eda/store_type_sales.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # Summary statistics
    # --------------------------------------------------

    summary = pd.DataFrame({
        "Metric": [
            "Total Sales",
            "Average Sales",
            "Median Sales",
            "Maximum Sales",
            "Minimum Sales",
            "Unique Stores"
        ],
        "Value": [
            df["Sales"].sum(),
            df["Sales"].mean(),
            df["Sales"].median(),
            df["Sales"].max(),
            df["Sales"].min(),
            df["Store"].nunique()
        ]
    })

    summary.to_csv(
        "outputs/eda/eda_summary.csv",
        index=False
    )

    print("\nEDA completed successfully.")

    print(
        "\nCharts saved to:",
        "outputs/eda/"
    )

    print(
        "\nSummary:"
    )

    print(summary)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_eda()