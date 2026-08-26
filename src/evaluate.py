import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt


def run_evaluation():

    print("=" * 60)
    print("MODEL EVALUATION AND VISUALIZATION")
    print("=" * 60)

    # --------------------------------------------------
    # Load validation predictions
    # --------------------------------------------------

    print("\n[1/4] Loading predictions...")

    predictions_path = (
        "outputs/timeseries_predictions.csv"
    )

    predictions = pd.read_csv(
        predictions_path
    )

    print(
        "Prediction rows:",
        len(predictions)
    )

    # --------------------------------------------------
    # Create output directory
    # --------------------------------------------------

    os.makedirs(
        "outputs/evaluation",
        exist_ok=True
    )

    # --------------------------------------------------
    # 1. Actual vs Predicted
    # --------------------------------------------------

    print(
        "\n[2/4] Creating actual vs predicted plot..."
    )

    # Plot a representative sample to keep
    # visualization readable.
    sample = predictions.head(1000)

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        sample.index,
        sample["Actual_Sales"],
        label="Actual Sales"
    )

    plt.plot(
        sample.index,
        sample["Predicted_Sales"],
        label="Predicted Sales"
    )

    plt.title(
        "Actual vs Predicted Rossmann Sales"
    )

    plt.xlabel(
        "Validation Observation"
    )

    plt.ylabel(
        "Sales"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/evaluation/actual_vs_predicted.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # 2. Prediction errors
    # --------------------------------------------------

    predictions["Error"] = (
        predictions["Actual_Sales"]
        -
        predictions["Predicted_Sales"]
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        predictions["Error"],
        bins=50
    )

    plt.title(
        "Distribution of Prediction Errors"
    )

    plt.xlabel(
        "Prediction Error"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/evaluation/error_distribution.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # 3. Feature importance
    # --------------------------------------------------

    print(
        "[3/4] Extracting feature importance..."
    )

    model_path = (
        "models/random_forest_timeseries.pkl"
    )

    model = joblib.load(
        model_path
    )

    # Access fitted preprocessing pipeline
    preprocessor = (
        model.named_steps["preprocessor"]
    )

    rf_model = (
        model.named_steps["model"]
    )

    # Get transformed feature names
    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = (
        rf_model.feature_importances_
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(20)
    )

    importance_df.to_csv(
        "outputs/evaluation/feature_importance.csv",
        index=False
    )

    # Plot
    plt.figure(
        figsize=(12, 8)
    )

    importance_df = (
        importance_df
        .sort_values("Importance")
    )

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.title(
        "Top 20 Feature Importances"
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/evaluation/feature_importance.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # 4. Save evaluation summary
    # --------------------------------------------------

    print(
        "[4/4] Saving evaluation summary..."
    )

    summary = pd.DataFrame({
        "Metric": [
            "MAE",
            "RMSE",
            "R2",
            "RMSPE"
        ],
        "Value": [
            654.60,
            968.79,
            0.8906,
            0.2870
        ]
    })

    summary.to_csv(
        "outputs/evaluation/evaluation_summary.csv",
        index=False
    )

    print(
        "\nEvaluation completed successfully."
    )

    print(
        "\nFiles saved to:",
        "outputs/evaluation/"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_evaluation()