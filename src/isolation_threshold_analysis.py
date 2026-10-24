from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import pandas as pd
import joblib
import numpy as np


TRAIN_FILE = str(PROJECT_ROOT / "data/campus_dns_features.csv")
MODEL_FILE = str(PROJECT_ROOT / "models/isolation_forest_model.pkl")


FEATURES = [
    "subdomain_length",
    "query_entropy",
    "inter_query_interval",
    "interval_std_10q",
    "queries_last_5s",
    "nxdomain_ratio_1s",
    "repeated_query_ratio_1s"
]


def main():

    print("Loading benign training data...")

    data = pd.read_csv(TRAIN_FILE)

    benign = data[data["label"] == "Benign"].copy()

    X_benign = benign[FEATURES]

    print("Benign training records:", len(X_benign))

    model = joblib.load(MODEL_FILE)

    print("\nCalculating Isolation Forest decision scores...")

    scores = model.decision_function(X_benign)

    print("\nBenign training score summary")
    print("-----------------------------")
    print(f"Minimum: {scores.min():.6f}")
    print(f"Maximum: {scores.max():.6f}")
    print(f"Mean:    {scores.mean():.6f}")
    print(f"Median:  {np.median(scores):.6f}")

    percentiles = [1, 2, 5, 10]

    print("\nLower-tail percentiles")
    print("----------------------")

    for p in percentiles:
        threshold = np.percentile(scores, p)

        predicted_anomalies = (scores < threshold).sum()

        print(
            f"{p:>2}% percentile threshold: "
            f"{threshold:.6f} | "
            f"Benign flagged: {predicted_anomalies}/{len(scores)}"
        )


if __name__ == "__main__":
    main()
