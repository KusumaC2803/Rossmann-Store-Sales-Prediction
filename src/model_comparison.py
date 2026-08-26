import os
import pandas as pd
import matplotlib.pyplot as plt


def run_model_comparison():

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    # --------------------------------------------------
    # Baseline model results
    # --------------------------------------------------

    baseline = {
        "Model": "Baseline Random Forest",
        "MAE": 2312.52,
        "RMSE": 3232.12,
        "R2": -0.2231,
        "RMSPE": 0.5959
    }

    # --------------------------------------------------
    # Improved time-series model results
    # --------------------------------------------------

    timeseries = {
        "Model": "Time-Series Random Forest",
        "MAE": 654.60,
        "RMSE": 968.79,
        "R2": 0.8906,
        "RMSPE": 0.2870
    }

    results = pd.DataFrame([
        baseline,
        timeseries
    ])

    # --------------------------------------------------
    # Create output directory
    # --------------------------------------------------

    os.makedirs(
        "outputs/comparison",
        exist_ok=True
    )

    # --------------------------------------------------
    # Save comparison table
    # --------------------------------------------------

    results.to_csv(
        "outputs/comparison/model_comparison.csv",
        index=False
    )

    # --------------------------------------------------
    # Calculate improvement
    # --------------------------------------------------

    mae_improvement = (
        (
            baseline["MAE"]
            -
            timeseries["MAE"]
        )
        /
        baseline["MAE"]
    ) * 100

    rmse_improvement = (
        (
            baseline["RMSE"]
            -
            timeseries["RMSE"]
        )
        /
        baseline["RMSE"]
    ) * 100

    rmspe_improvement = (
        (
            baseline["RMSPE"]
            -
            timeseries["RMSPE"]
        )
        /
        baseline["RMSPE"]
    ) * 100

    r2_improvement = (
        timeseries["R2"]
        -
        baseline["R2"]
    )

    improvement = pd.DataFrame({
        "Metric": [
            "MAE Reduction (%)",
            "RMSE Reduction (%)",
            "RMSPE Reduction (%)",
            "R2 Improvement"
        ],
        "Improvement": [
            mae_improvement,
            rmse_improvement,
            rmspe_improvement,
            r2_improvement
        ]
    })

    improvement.to_csv(
        "outputs/comparison/model_improvement.csv",
        index=False
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\nModel Performance:")
    print(results.to_string(index=False))

    print("\nImprovement:")
    print(improvement.to_string(index=False))

    # --------------------------------------------------
    # Create comparison chart
    # --------------------------------------------------

    print(
        "\nCreating model comparison chart..."
    )

    metrics = [
        "MAE",
        "RMSE",
        "RMSPE"
    ]

    baseline_values = [
        baseline["MAE"],
        baseline["RMSE"],
        baseline["RMSPE"] * 10000
    ]

    timeseries_values = [
        timeseries["MAE"],
        timeseries["RMSE"],
        timeseries["RMSPE"] * 10000
    ]

    # Normalize metrics so they can be compared
    # visually on the same scale.
    normalized_baseline = []
    normalized_timeseries = []

    for b, t in zip(
        baseline_values,
        timeseries_values
    ):
        maximum = max(b, t)

        normalized_baseline.append(
            (b / maximum) * 100
        )

        normalized_timeseries.append(
            (t / maximum) * 100
        )

    x = range(len(metrics))
    width = 0.35

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        [i - width / 2 for i in x],
        normalized_baseline,
        width=width,
        label="Baseline Random Forest"
    )

    plt.bar(
        [i + width / 2 for i in x],
        normalized_timeseries,
        width=width,
        label="Time-Series Random Forest"
    )

    plt.xticks(
        list(x),
        metrics
    )

    plt.ylabel(
        "Relative Score (%)"
    )

    plt.title(
        "Baseline vs Time-Series Model"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/comparison/model_comparison.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # Completion
    # --------------------------------------------------

    print(
        "\nComparison completed successfully."
    )

    print(
        "Files saved to:",
        "outputs/comparison/"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_model_comparison()