import pandas as pd


def load_data():
    train = pd.read_csv(
        "data/train.csv",
        low_memory=False
    )

    store = pd.read_csv(
        "data/store.csv",
        low_memory=False
    )

    return train, store


def preprocess_data(train, store):

    # Merge store information
    df = train.merge(
        store,
        on="Store",
        how="left"
    )

    # Convert date
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort chronologically by store
    df = df.sort_values(
        ["Store", "Date"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Handle missing values
    # --------------------------------------------------

    df["CompetitionDistance"] = df[
        "CompetitionDistance"
    ].fillna(
        df["CompetitionDistance"].median()
    )

    numeric_columns = [
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear",
    ]

    for column in numeric_columns:
        df[column] = df[column].fillna(0)

    df["PromoInterval"] = df[
        "PromoInterval"
    ].fillna("None")

    # --------------------------------------------------
    # Fix mixed data types
    # --------------------------------------------------

    # StateHoliday contains values such as:
    # 0, "0", "a", "b", "c"
    # Convert it consistently to string.
    df["StateHoliday"] = df[
        "StateHoliday"
    ].astype(str)

    # Other categorical columns
    categorical_columns = [
        "StoreType",
        "Assortment",
        "PromoInterval",
    ]

    for column in categorical_columns:
        df[column] = df[column].astype(str)

    return df