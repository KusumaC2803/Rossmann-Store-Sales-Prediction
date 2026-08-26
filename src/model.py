from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


def build_model(X):

    # Identify categorical columns
    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # Identify numerical columns
    numerical_columns = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                ),
                categorical_columns,
            ),
            (
                "numerical",
                "passthrough",
                numerical_columns,
            ),
        ]
    )

    # Random Forest regression model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    # Complete ML pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline